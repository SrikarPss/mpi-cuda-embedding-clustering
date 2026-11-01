#include "ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#ifdef HPC_OPENMP
#include <omp.h>
#endif

namespace {
thread_local std::string last_error;
int workers(int requested, bool parallel) {
#ifdef HPC_OPENMP
    omp_set_dynamic(0);
    return parallel ? std::max(1, requested) : 1;
#else
    if (parallel) throw std::runtime_error("OpenMP was not compiled into this library");
    return 1;
#endif
}
float dot(const float* a, const float* b, int d) {
    float sum = 0;
#pragma omp simd reduction(+:sum)
    for (int j = 0; j < d; ++j) sum += a[j] * b[j];
    return sum;
}
template<class F> int protect(F operation) {
    try { last_error.clear(); operation(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}

namespace hpc {
void assign(const float* x, const float* centers, int* labels, int n, int dim,
            int clusters, bool parallel) {
#pragma omp parallel for if(parallel) schedule(static)
    for (int i = 0; i < n; ++i) {
        float best = -std::numeric_limits<float>::infinity();
        int label = 0;
        for (int c = 0; c < clusters; ++c) {
            float score = dot(x + size_t(i)*dim, centers + size_t(c)*dim, dim);
            if (score > best) { best = score; label = c; }
        }
        labels[i] = label;
    }
}
void accumulate(const float* x, const int* labels, float* sums, int* counts,
                int n, int dim, int clusters, bool parallel) {
    int nt = 1;
#ifdef HPC_OPENMP
    nt = parallel ? omp_get_max_threads() : 1;
#endif
    size_t stride = size_t(clusters)*dim;
    std::vector<float> local_sums(size_t(nt)*stride, 0);
    std::vector<int> local_counts(size_t(nt)*clusters, 0);
#pragma omp parallel if(parallel)
    {
        int tid = 0;
#ifdef HPC_OPENMP
        tid = omp_get_thread_num();
#endif
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            int c = labels[i];
            ++local_counts[size_t(tid)*clusters + c];
            float* dest = local_sums.data() + size_t(tid)*stride + size_t(c)*dim;
#pragma omp simd
            for (int j = 0; j < dim; ++j) dest[j] += x[size_t(i)*dim+j];
        }
    }
    std::fill(sums, sums + stride, 0);
    std::fill(counts, counts + clusters, 0);
    for (int t = 0; t < nt; ++t) {
        for (size_t j = 0; j < stride; ++j) sums[j] += local_sums[size_t(t)*stride+j];
        for (int c = 0; c < clusters; ++c) counts[c] += local_counts[size_t(t)*clusters+c];
    }
}
void update(float* centers, const float* sums, const int* counts, int dim, int clusters) {
    for (int c = 0; c < clusters; ++c) {
        if (!counts[c]) continue; // Preserve the previous center for an empty cluster.
        double norm = 0;
        for (int j = 0; j < dim; ++j) norm += double(sums[size_t(c)*dim+j])*sums[size_t(c)*dim+j];
        if (norm <= 1e-30) continue;
        float inv = float(1 / std::sqrt(norm));
        for (int j = 0; j < dim; ++j) centers[size_t(c)*dim+j] = sums[size_t(c)*dim+j]*inv;
    }
}
}

extern "C" const char* hpc_error() { return last_error.c_str(); }

extern "C" int hpc_has_openmp() {
#ifdef HPC_OPENMP
    return 1;
#else
    return 0;
#endif
}

extern "C" int hpc_kmeans(const float* x, float* centers, int* labels, int n, int dim,
                          int clusters, int iterations, int threads, int parallel, double* objectives) {
    return protect([&] {
        if (n < 1 || dim < 1 || clusters < 1 || clusters > n || iterations < 1 || threads < 1)
            throw std::invalid_argument("Invalid clustering shape or iteration count");
        int nt = workers(threads, parallel);
#ifdef HPC_OPENMP
        omp_set_num_threads(nt);
#endif
        std::vector<float> sums(size_t(clusters)*dim);
        std::vector<int> counts(clusters);
        for (int it = 0; it <= iterations; ++it) {
            hpc::assign(x, centers, labels, n, dim, clusters, parallel);
            double objective = 0;
#pragma omp parallel for num_threads(nt) if(parallel) reduction(+:objective)
            for (int i = 0; i < n; ++i)
                objective += 1.0 - dot(x+size_t(i)*dim, centers+size_t(labels[i])*dim, dim);
            objectives[it] = objective;
            if (it == iterations) break;
            hpc::accumulate(x, labels, sums.data(), counts.data(), n, dim, clusters, parallel);
            hpc::update(centers, sums.data(), counts.data(), dim, clusters);
        }
    });
}
