#!/usr/bin/env bash
# Activate the same runtime on all nodes before submission; pass account/partition to sbatch.
#SBATCH --job-name=recap_browsecomp_plus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_DIR=${RECAP_ROOT:-${SLURM_SUBMIT_DIR}}
export RECAP_ENV=browsecomp_plus
exec bash "${PROJECT_DIR}/scripts/run_slurm_ray.sh" \
    bash "${PROJECT_DIR}/scripts/browsecomp_plus/train/train.sh" "$@"
