#!/usr/bin/env bash
# Run one command on a Ray head, using every node in the current Slurm allocation.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export EXPECTED_RAY_NODES=${SLURM_JOB_NUM_NODES:?Run this helper inside a Slurm allocation}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export PYTHONUNBUFFERED=1
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
srun_args=(--overlap --nodes=1 --ntasks=1 --cpu-bind=none --export=ALL)
if [[ -n "${SLURM_ENVIRONMENT:-}" ]]; then
    srun_args+=(--environment="${SLURM_ENVIRONMENT}")
fi
# Quote the caller's arguments once before sending the command to the head node.
printf -v run_command '%q ' "$@"
export RECAP_RUN_COMMAND=${run_command}
export RECAP_PROJECT_DIR=${PROJECT_DIR}
export RAY_PORT=${RAY_PORT:-6379}
head_ip=$(srun "${srun_args[@]}" -w "${nodes[0]}" hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /:/){print $i; exit}}')
export RAY_ADDRESS=${head_ip}:${RAY_PORT}
export RAY_HEAD_IP=${head_ip}
cleanup() {
    status=$?
    trap - EXIT TERM INT
    for node in "${nodes[@]}"; do
        srun "${srun_args[@]}" -w "${node}" python -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
    done
    if [[ "${RECAP_ENV:-}" == "swe_agent" ]]; then
        srun "${srun_args[@]}" -w "${nodes[0]}" timeout 120s python "${PROJECT_DIR}/agent_system/envs/swe_agent/e2b_episode.py" --cleanup-job || true
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun "${srun_args[@]}" -w "${nodes[0]}" bash -c '
    set -euo pipefail
    python -m ray.scripts.scripts start --head --node-ip-address="${RAY_HEAD_IP}" --port="${RAY_PORT}" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats
    cd "${RECAP_PROJECT_DIR}"
    python scripts/wait_for_ray_nodes.py
    eval "exec ${RECAP_RUN_COMMAND}"
' &
driver=$!
sleep 5
for node in "${nodes[@]:1}"; do
    srun "${srun_args[@]}" -w "${node}" bash -c '
        exec python -m ray.scripts.scripts start --address="${RAY_ADDRESS}" \
            --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats --block
    ' &
done
wait "${driver}"
