#!/bin/bash
# Pass your allocation at submission time with sbatch --account=...
#SBATCH --partition=compute
#SBATCH --gpus-per-node=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=05:00:00
#SBATCH --job-name=nnsyn-300ep
#SBATCH --output=nnsyn_300ep_%j.out
set -euo pipefail

: "${NNSYN_BASE:?Set NNSYN_BASE to the run directory}"
: "${SYNTHRAD_AB_ROOT:?Set SYNTHRAD_AB_ROOT to the extracted Task1/AB folder}"
: "${COURSE_REPO:?Set COURSE_REPO to this mini-course checkout}"

module purge
module load python/3.11.5
source "$NNSYN_BASE/env/bin/activate"
export nnUNet_n_proc_DA=0
export nnUNet_def_n_proc=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$COURSE_REPO/logs"

python "$COURSE_REPO/code/full_run/run_full_baseline.py" train \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT" --fold 0
