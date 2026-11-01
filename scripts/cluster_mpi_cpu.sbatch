#!/usr/bin/env bash
#SBATCH --job-name=cluster-mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=cluster-mpi-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this repository root}"
task_python="${HPC_PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
task_out="results/mpi-${SLURM_JOB_ID}"
"$task_python" scripts/build.py --no-cuda
"$task_python" run.py --prepare-only --rows 200000 --dim 128 \
  --clusters 32 --iterations 20 --output "$task_out"
mpirun -n "${SLURM_NTASKS:-2}" --map-by ppr:1:node --bind-to none build/hpc_cluster_mpi \
  --input "$task_out/points.f32" --initial "$task_out/initial.f32" \
  --rows 200000 --dim 128 --clusters 32 --iterations 20 --threads "$OMP_NUM_THREADS" \
  --backend openmp --output "$task_out/distributed"
