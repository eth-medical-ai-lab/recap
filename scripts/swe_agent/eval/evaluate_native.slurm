#!/usr/bin/env bash
# Usage: sbatch evaluate_native.slurm /absolute/path/to/config.json
#SBATCH --job-name=swe_native
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_DIR=${RECAP_ROOT:-${SLURM_SUBMIT_DIR}}
CONFIG=${1:?Pass a native evaluation config.json}
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
: "${E2B_API_KEY:?Export your E2B API key}"
cd "${PROJECT_DIR}"
srun_args=(--cpu-bind=none --export=ALL)
if [[ -n "${SLURM_ENVIRONMENT:-}" ]]; then
    srun_args+=(--environment="${SLURM_ENVIRONMENT}")
fi
cleanup() {
    srun "${srun_args[@]}" --nodes=1 --ntasks=1 timeout 120s python agent_system/envs/swe_agent/e2b_episode.py --cleanup-job || true
}
trap cleanup EXIT
srun "${srun_args[@]}" --nodes=1 --ntasks=1 python -m scripts.swe_agent.eval.evaluate_native "${CONFIG}" --prepare
srun "${srun_args[@]}" --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 python -m scripts.swe_agent.eval.evaluate_native "${CONFIG}"
srun "${srun_args[@]}" --nodes=1 --ntasks=1 python -m scripts.swe_agent.eval.evaluate_native "${CONFIG}" --summarize
