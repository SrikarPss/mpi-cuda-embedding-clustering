#include "ops.h"
#include <mpi.h>
#ifdef HPC_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template<class T> void read_raw(const std::string& path,std::vector<T>& values) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(values.size()*sizeof(T))) throw std::runtime_error("Wrong file size: "+path);
    file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(T));
    if(!file) throw std::runtime_error("Cannot read "+path);
}
template<class T> void write_raw(const std::string& path,const std::vector<T>& values) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    if(!file) throw std::runtime_error("Cannot write "+path);
}
}
int main(int argc,char** argv) {
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        std::map<std::string,std::string> options;
        if((argc-1)%2) throw std::runtime_error("Arguments must be --name value pairs");
        for(int i=1;i<argc;i+=2) options[argv[i]]=argv[i+1];
        auto get=[&](const char* name) {
            auto item=options.find(name);
            if(item==options.end()) throw std::runtime_error(std::string("Missing ")+name);
            return item->second;
        };
        int n=std::stoi(get("--rows")),d=std::stoi(get("--dim")),k=std::stoi(get("--clusters"));
        int iterations=std::stoi(get("--iterations")),threads=std::stoi(get("--threads"));
        std::string backend=get("--backend"),output=get("--output");
        bool parallel=backend=="openmp";
        bool gpu=backend=="cuda-naive" || backend=="cuda-optimized" || backend=="cublas";
        if(!gpu && !parallel && backend!="cpu") throw std::runtime_error("Unknown backend");
        if(n<1 || d<1 || k<1 || k>n || iterations<1 || threads<1 ||
           size_t(n)*d>INT_MAX || size_t(k)*d>INT_MAX) throw std::runtime_error("Invalid shape or MPI int-count limit exceeded");
#ifdef HPC_OPENMP
        omp_set_dynamic(0);omp_set_num_threads(parallel?threads:1);
#else
        if(parallel) throw std::runtime_error("OpenMP is not compiled in");
#endif
        int gpu_mode=backend=="cuda-naive"?0:(backend=="cublas"?2:1);
        if(gpu) {
#ifdef HPC_CUDA
            int devices=hpc_cuda_devices();
            if(!devices) throw std::runtime_error(hpc_cuda_error());
            MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);
            int local_rank;MPI_Comm_rank(local,&local_rank);MPI_Comm_free(&local);
            if(local_rank>=devices) throw std::runtime_error("Use at most one GPU worker per local GPU");
            if(hpc_cuda_set_device(local_rank)) throw std::runtime_error(hpc_cuda_error());
#else
            throw std::runtime_error("CUDA backend was not compiled in");
#endif
        }
        std::vector<int> rows(size),elements(size),offsets(size),row_offsets(size);
        int cursor=0;
        for(int r=0;r<size;++r) {
            rows[r]=n/size+(r<n%size);row_offsets[r]=cursor;
            elements[r]=rows[r]*d;offsets[r]=cursor*d;cursor+=rows[r];
        }
        std::vector<float> all(rank==0?size_t(n)*d:0),centers(size_t(k)*d);
        if(!rank) {read_raw(get("--input"),all);read_raw(get("--initial"),centers);}
        MPI_Bcast(centers.data(),k*d,MPI_FLOAT,0,MPI_COMM_WORLD);
        int local_n=rows[rank];std::vector<float> x(size_t(local_n)*d);
        MPI_Scatterv(all.data(),elements.data(),offsets.data(),MPI_FLOAT,x.data(),local_n*d,MPI_FLOAT,0,MPI_COMM_WORLD);
        std::vector<int> labels(local_n),counts(k),global_counts(k);
        std::vector<float> sums(size_t(k)*d),global_sums(size_t(k)*d);
        std::vector<double> objectives(iterations+1);
        double compute=0,communication=0;
        MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();
        for(int it=0;it<=iterations;++it) {
            double t=MPI_Wtime();
            if(gpu) {
#ifdef HPC_CUDA
                if(hpc_cuda_step(x.data(),centers.data(),labels.data(),sums.data(),counts.data(),local_n,d,k,gpu_mode))
                    throw std::runtime_error(hpc_cuda_error());
#endif
            } else {
                hpc::assign(x.data(),centers.data(),labels.data(),local_n,d,k,parallel);
                if(it<iterations) hpc::accumulate(x.data(),labels.data(),sums.data(),counts.data(),local_n,d,k,parallel);
            }
            double local_objective=0;
#pragma omp parallel for if(parallel) reduction(+:local_objective)
            for(int i=0;i<local_n;++i) {
                float dot=0;
#pragma omp simd reduction(+:dot)
                for(int a=0;a<d;++a) dot+=x[size_t(i)*d+a]*centers[size_t(labels[i])*d+a];
                local_objective+=1.0-dot;
            }
            compute+=MPI_Wtime()-t;t=MPI_Wtime();
            MPI_Allreduce(&local_objective,&objectives[it],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            if(it<iterations) {
                MPI_Allreduce(sums.data(),global_sums.data(),k*d,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
                MPI_Allreduce(counts.data(),global_counts.data(),k,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
            }
            communication+=MPI_Wtime()-t;
            if(it==iterations) break;
            t=MPI_Wtime();hpc::update(centers.data(),global_sums.data(),global_counts.data(),d,k);compute+=MPI_Wtime()-t;
        }
        double elapsed=MPI_Wtime()-start,max_elapsed,max_compute,max_communication;
        MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&compute,&max_compute,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&communication,&max_communication,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        std::vector<int> all_labels(rank==0?n:0);
        MPI_Gatherv(labels.data(),local_n,MPI_INT,all_labels.data(),rows.data(),row_offsets.data(),MPI_INT,0,MPI_COMM_WORLD);
        if(!rank) {
            write_raw(output+".centers.f32",centers);write_raw(output+".labels.i32",all_labels);
            std::ofstream file(output+".json");
            file.precision(12);
            file<<"{\"backend\":\""<<backend<<"\",\"ranks\":"<<size<<",\"threads_per_rank\":"<<threads
                <<",\"iterations\":"<<iterations<<",\"rows\":"<<n<<",\"dim\":"<<d<<",\"clusters\":"<<k
                <<",\"compute_loop_seconds\":"<<max_elapsed<<",\"max_rank_compute_seconds\":"<<max_compute
                <<",\"max_rank_collective_seconds\":"<<max_communication<<",\"objectives\":[";
            for(int i=0;i<=iterations;++i) file<<(i?",":"")<<objectives[i];
            file<<"]}\n";if(!file) throw std::runtime_error("Cannot write metadata");
            std::cout<<"MPI clustering: ranks="<<size<<" objective="<<objectives.back()<<" seconds="<<max_elapsed<<"\n";
        }
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Rank "<<rank<<": "<<e.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
