#!/usr/bin/env bash
# Usage: sbatch evaluate.slurm evaluation.model_path=/path/to/model evaluation.output_dir=/path/to/run
#SBATCH --job-name=recap_browsecomp_plus_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_DIR=${RECAP_ROOT:-${SLURM_SUBMIT_DIR}}
export RECAP_ENV=browsecomp_plus
exec bash "${PROJECT_DIR}/scripts/run_slurm_ray.sh" \
    python -m scripts.browsecomp_plus.eval.evaluate "$@"
