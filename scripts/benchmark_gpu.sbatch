#!/usr/bin/env bash
#SBATCH --job-name=distributed-clus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=distributed-clustering-gpu-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this repository root}"
# Activate a Linux Python environment and load the cluster CUDA module before submission.
task_python="${HPC_PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi
"$task_python" scripts/build.py --no-mpi
"$task_python" -c 'import sys; sys.path.insert(0,"src"); from distributed_clustering.runtime import require; require("cuda-naive"); require("cuda-optimized"); require("cublas")'
"$task_python" -m unittest discover -s tests -v
"$task_python" run.py --backends cpu openmp cuda-naive cuda-optimized cublas \
  --threads "$OMP_NUM_THREADS" --repeats 5 --output "results/gpu-${SLURM_JOB_ID}"
