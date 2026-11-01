# Distributed embedding clustering and outlier discovery

Partition normalized embeddings with spherical k-means, select representative items, and rank points by cosine distance from their assigned center.

## Single-process and local MPI runs

From this repository root, with its virtual environment activated and native library built:

```bash
python run.py --rows 8000 --dim 64 --clusters 12 \
  --iterations 15 --backends cpu openmp mps --mpi-ranks 1 2 4
```

Use `--input-file /path/to/embeddings.npy` for real data. `--initial-file centers.npy` supplies initial centers; otherwise seeded cosine-distance sampling chooses a common initialization. For the built-in demo, `--outlier-fraction` controls the fraction of uniform random points inserted into concentrated spherical clusters.

Assignment maximizes dot product. Updating a center normalizes its member sum; an empty cluster or a sum with negligible norm retains its previous center. The objective is `sum(1 - dot(point, assigned_center))`. After the requested number of updates, labels are reassigned to the final centers. Seeded initialization is shared across implementations, but parallel floating-point summation may change close assignments. Quality checks validate unit centers, objective consistency/monotonicity, and nearest-center assignments rather than requiring identical cluster-size distributions.

CPU/OpenMP workers use private per-thread centroid sums followed by a reduction. Naive CUDA performs global floating-point atomics per feature; optimized CUDA accumulates cluster/feature tiles in shared memory before global merging. cuBLAS is an additional assignment-score baseline. MPS uses tensor matrix multiplication and indexed accumulation.

## Hybrid MPI

The native `build/hpc_cluster_mpi` executable uses uneven `MPI_Scatterv` partitions, broadcasts the common initialization, and alternates local assignment/reduction with global `MPI_Allreduce` of centroid sums/counts. It gathers final labels using `MPI_Gatherv`. Empty local partitions are supported. MPI is initialized with `MPI_THREAD_FUNNELED`; only the main thread makes MPI calls.

`--mpi-ranks` launches local MPI jobs and includes launch, scatter, and output time in its wall-clock measurements. `--mpi-hosts host1,host2` can route ranks to remote hosts with a shared filesystem and matching environment. Prefer Slurm templates for scheduled clusters. Networked execution has not been validated on the development Mac.

Generate raw inputs without running benchmarks:

```bash
python run.py --prepare-only --rows 8000 --dim 64 \
  --clusters 12 --iterations 15 --output results/mpi-input
```

Then launch the native program:

```bash
mpirun --oversubscribe -n 2 build/hpc_cluster_mpi \
  --input results/mpi-input/points.f32 --initial results/mpi-input/initial.f32 \
  --rows 8000 --dim 64 --clusters 12 --iterations 15 --threads 4 \
  --backend openmp --output results/mpi-input/two-ranks
```

Files contain native-endian row-major float32 values. Labels are int32. Shapes and file sizes are checked; the MPI executable expects normalized finite embeddings prepared by the Python loader. Total input element counts must fit MPI's int32 count/displacement interface. Native `--backend` supports `cpu`, `openmp`, `cuda-naive`, `cuda-optimized`, and `cublas`; GPU variants require a CUDA-enabled build and at most one worker per visible local GPU. `--threads` controls OpenMP workers per rank.

For multi-node CPU and GPU examples, see `scripts/cluster_mpi_cpu.sbatch` and `scripts/cluster_mpi_gpu.sbatch`. Hold problem size fixed for strong scaling; increase rows proportionally to rank count for weak scaling. The JSON records max-rank compute-loop and collective time; those scopes differ from end-to-end launcher wall time.

## Outputs and validation

`results/local/` contains initial/input data, per-backend centers/labels/objective histories, cluster representative CSVs, ranked outlier CSVs, convergence plots, and timing JSON/CSV. Synthetic demo quality includes inlier label accuracy after Hungarian matching and recall among the top known number of injected outliers. Unknown real-data quality is not inferred from synthetic labels.

```bash
python scripts/sweep.py --sizes 1000 8000 32000 --threads 1 2 4
HPC_TEST_MPI=1 python -m unittest discover -s tests -v
```

The opt-in MPI tests compare against single-process results for uneven partitions and for more ranks than data points. Run them from a regular terminal; the restricted development session blocks MPI socket creation.
