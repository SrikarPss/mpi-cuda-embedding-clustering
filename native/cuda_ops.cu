#include "ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const T* data, size_t n) { if (n) check(cudaMemcpy(p,data,n*sizeof(T),cudaMemcpyHostToDevice)); }
    void get(T* data, size_t n) const { if (n) check(cudaMemcpy(data,p,n*sizeof(T),cudaMemcpyDeviceToHost)); }
    void zero(size_t n) { if (n) check(cudaMemset(p,0,n*sizeof(T))); }
};
struct Blas {
    cublasHandle_t handle = nullptr;
    explicit Blas(bool needed) {
        if (needed) {
            blas_check(cublasCreate(&handle));
            auto status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            if (status != CUBLAS_STATUS_SUCCESS) { cublasDestroy(handle); handle=nullptr; blas_check(status); }
        }
    }
    ~Blas() { if (handle) cublasDestroy(handle); }
};
template<class F> int protect(F f) {
    try { error_text.clear(); f(); check(cudaGetLastError()); check(cudaDeviceSynchronize()); return 0; }
    catch (const std::exception& e) { error_text=e.what(); return -1; }
}
__global__ void similarity_basic(const float* q,const float* db,float* s,int nq,int n,int d) {
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p>=size_t(nq)*n) return;
    int i=int(p/n), j=int(p%n);
    float v=0;
    for(int a=0;a<d;++a) v+=q[size_t(i)*d+a]*db[size_t(j)*d+a];
    s[p]=v;
}

__global__ void similarity_tiled(const float* q,const float* db,float* s,int nq,int n,int d) {
    __shared__ float a[16][16],b[16][17];
    int x=threadIdx.x,y=threadIdx.y;
    int row=blockIdx.y*16+y,col=blockIdx.x*16+x;
    float value=0;
    for(int t=0;t<d;t+=16) {
        a[y][x]=(row<nq && t+x<d)?q[size_t(row)*d+t+x]:0;
        b[y][x]=(col<n && t+y<d)?db[size_t(col)*d+t+y]:0;
        __syncthreads();
        #pragma unroll
        for(int j=0;j<16;++j) value+=a[y][j]*b[j][x];
        __syncthreads();
    }
    if(row<nq && col<n) s[size_t(row)*n+col]=value;
}

void similarity(const float* q,const float* db,float* s,int nq,int n,int d,int mode,Blas& blas) {
    if(mode==2) {
        float alpha=1,beta=0;
        blas_check(cublasSgemm(blas.handle,CUBLAS_OP_T,CUBLAS_OP_N,n,nq,d,&alpha,db,d,q,d,&beta,s,n));
    } else if(mode==1) {
        similarity_tiled<<<dim3((n+15)/16,(nq+15)/16),dim3(16,16)>>>(q,db,s,nq,n,d);
    } else {
        similarity_basic<<<unsigned((size_t(nq)*n+255)/256),256>>>(q,db,s,nq,n,d);
    }
    check(cudaGetLastError());
}

__global__ void assign_basic(const float* x,const float* c,int* labels,int n,int d,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float best=-INFINITY; int label=0;
    for(int j=0;j<k;++j) {
        float sum=0;
        for(int a=0;a<d;++a) sum+=x[size_t(i)*d+a]*c[size_t(j)*d+a];
        if(sum>best) {best=sum; label=j;}
    }
    labels[i]=label;
}

__global__ void assign_scores(const float* scores,int* labels,int n,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float best=-INFINITY; int label=0;
    for(int j=0;j<k;++j) if(scores[size_t(i)*k+j]>best) {best=scores[size_t(i)*k+j];label=j;}
    labels[i]=label;
}

__global__ void accumulate_basic(const float* x,const int* labels,float* sums,int* counts,int n,int d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int c=labels[i]; atomicAdd(counts+c,1);
    for(int j=0;j<d;++j) atomicAdd(sums+size_t(c)*d+j,x[size_t(i)*d+j]);
}

__global__ void accumulate_shared(const float* x,const int* labels,float* sums,int* counts,
                                  int n,int d,int k,int feature_tile) {
    extern __shared__ float local[];
    int t=threadIdx.x,i=blockIdx.x*blockDim.x+t,first=blockIdx.y*feature_tile;
    for(int j=t;j<k*feature_tile;j+=blockDim.x) local[j]=0;
    __syncthreads();
    if(i<n) {
        int c=labels[i];
        if(!blockIdx.y) atomicAdd(counts+c,1);
        for(int j=0;j<feature_tile && first+j<d;++j)
            atomicAdd(local+c*feature_tile+j,x[size_t(i)*d+first+j]);
    }
    __syncthreads();
    for(int j=t;j<k*feature_tile;j+=blockDim.x) {
        int c=j/feature_tile,a=first+j%feature_tile;
        if(a<d && local[j]!=0) atomicAdd(sums+size_t(c)*d+a,local[j]);
    }
}

__global__ void update_centers(float* centers,const float* sums,const int* counts,int d,int k) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=k || !counts[c]) return;
    double norm=0;
    for(int j=0;j<d;++j) { double v=sums[size_t(c)*d+j]; norm+=v*v; }
    if(norm<=1e-30) return;
    float inv=float(1/sqrt(norm));
    for(int j=0;j<d;++j) centers[size_t(c)*d+j]=sums[size_t(c)*d+j]*inv;
}

__global__ void objective_parts(const float* x,const float* c,const int* labels,double* parts,int n,int d) {
    __shared__ double local[256];
    int i=blockIdx.x*256+threadIdx.x,t=threadIdx.x;
    float dot=0;
    if(i<n) for(int j=0;j<d;++j) dot+=x[size_t(i)*d+j]*c[size_t(labels[i])*d+j];
    local[t]=(i<n)?1.0-double(dot):0;
    __syncthreads();
    for(int step=128;step;step/=2) {if(t<step) local[t]+=local[t+step];__syncthreads();}
    if(!t) parts[blockIdx.x]=local[0];
}

void assignment(const float* x,const float* c,int* labels,float* scores,int n,int d,int k,int mode,Blas& blas) {
    if(!n) return;
    if(mode==0) assign_basic<<<(n+255)/256,256>>>(x,c,labels,n,d,k);
    else { similarity(x,c,scores,n,k,d,mode,blas); assign_scores<<<(n+255)/256,256>>>(scores,labels,n,k); }
    check(cudaGetLastError());
}

void accumulation(const float* x,const int* labels,float* sums,int* counts,int n,int d,int k,int mode) {
    check(cudaMemset(sums,0,size_t(k)*d*sizeof(float)));
    check(cudaMemset(counts,0,size_t(k)*sizeof(int)));
    if(!n) return;
    if(!mode) accumulate_basic<<<(n+255)/256,256>>>(x,labels,sums,counts,n,d);
    else {
        cudaDeviceProp prop;int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
        int tile=std::min(32,int(prop.sharedMemPerBlock/(sizeof(float)*size_t(k))));
        if(tile<1) throw std::runtime_error("Too many clusters for shared-memory accumulation");
        accumulate_shared<<<dim3((n+127)/128,(d+tile-1)/tile),128,size_t(k)*tile*sizeof(float)>>>(x,labels,sums,counts,n,d,k,tile);
    }
    check(cudaGetLastError());
}

void validate_cluster(int n,int d,int k,int mode) {
    if(n<0 || d<1 || k<1 || mode<0 || mode>2) throw std::invalid_argument("Invalid CUDA clustering arguments");
}
}

extern "C" const char* hpc_cuda_error() {return error_text.c_str();}

extern "C" int hpc_cuda_devices() {int count=0;auto s=cudaGetDeviceCount(&count);if(s!=cudaSuccess) {error_text=cudaGetErrorString(s);return 0;}return count;}

extern "C" int hpc_cuda_set_device(int device) {return protect([&]{check(cudaSetDevice(device));});}

extern "C" int hpc_cuda_kmeans(const float* x,float* centers,int* labels,int n,int d,int k,int iterations,int mode,double* objectives) {
    return protect([&]{
        validate_cluster(n,d,k,mode);
        if(n<1 || k>n || iterations<1) throw std::invalid_argument("Invalid CUDA k-means size or iterations");
        Buffer<float> data(size_t(n)*d),c(size_t(k)*d),sums(size_t(k)*d),scores(mode?size_t(n)*k:0);
        Buffer<int> lab(n),counts(k);int blocks=(n+255)/256;Buffer<double> parts(blocks);
        std::vector<double> host_parts(blocks);Blas blas(mode==2);
        data.put(x,size_t(n)*d);c.put(centers,size_t(k)*d);
        for(int it=0;it<=iterations;++it) {
            assignment(data.p,c.p,lab.p,scores.p,n,d,k,mode,blas);
            objective_parts<<<blocks,256>>>(data.p,c.p,lab.p,parts.p,n,d);
            parts.get(host_parts.data(),blocks);objectives[it]=std::accumulate(host_parts.begin(),host_parts.end(),0.0);
            if(it==iterations) break;
            accumulation(data.p,lab.p,sums.p,counts.p,n,d,k,mode);
            update_centers<<<(k+127)/128,128>>>(c.p,sums.p,counts.p,d,k);
        }
        c.get(centers,size_t(k)*d);lab.get(labels,n);
    });
}

extern "C" int hpc_cuda_step(const float* x,const float* centers,int* labels,float* sums,int* counts,int n,int d,int k,int mode) {
    return protect([&]{
        validate_cluster(n,d,k,mode);
        Buffer<float> data(size_t(n)*d),c(size_t(k)*d),s(size_t(k)*d),scores(mode?size_t(n)*k:0);
        Buffer<int> lab(n),ct(k);Blas blas(mode==2);
        data.put(x,size_t(n)*d);c.put(centers,size_t(k)*d);
        assignment(data.p,c.p,lab.p,scores.p,n,d,k,mode,blas);
        accumulation(data.p,lab.p,s.p,ct.p,n,d,k,mode);
        lab.get(labels,n);s.get(sums,size_t(k)*d);ct.get(counts,k);
    });
}
