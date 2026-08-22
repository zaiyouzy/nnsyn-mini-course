#!/bin/bash
# Pass your allocation at submission time with sbatch --account=...
#SBATCH --partition=compute
#SBATCH --gpus-per-node=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=nnsyn-evaluate
#SBATCH --output=nnsyn_evaluate_%j.out
set -euo pipefail

: "${NNSYN_BASE:?Set NNSYN_BASE to the run directory}"
: "${SYNTHRAD_AB_ROOT:?Set SYNTHRAD_AB_ROOT to the extracted Task1/AB folder}"
: "${COURSE_REPO:?Set COURSE_REPO to this mini-course checkout}"

module purge
module load python/3.11.5
source "$NNSYN_BASE/env/bin/activate"
mkdir -p "$COURSE_REPO/logs"

python "$COURSE_REPO/code/full_run/run_full_baseline.py" evaluate \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT" --fold 0
