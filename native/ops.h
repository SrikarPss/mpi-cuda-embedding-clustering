#pragma once
#include <cstddef>
namespace hpc {
void assign(const float* x, const float* centers, int* labels, int n, int dim,
            int clusters, bool parallel);
void accumulate(const float* x, const int* labels, float* sums, int* counts,
                int n, int dim, int clusters, bool parallel);
void update(float* centers, const float* sums, const int* counts, int dim, int clusters);
}

extern "C" {
const char* hpc_error();
int hpc_has_openmp();
int hpc_kmeans(const float* x, float* centers, int* labels, int n, int dim,
               int clusters, int iterations, int threads, int parallel, double* objectives);
const char* hpc_cuda_error();
int hpc_cuda_devices();
int hpc_cuda_set_device(int device);
int hpc_cuda_kmeans(const float*, float*, int*, int, int, int, int, int, double*);
int hpc_cuda_step(const float*, const float*, int*, float*, int*, int, int, int, int);
}
