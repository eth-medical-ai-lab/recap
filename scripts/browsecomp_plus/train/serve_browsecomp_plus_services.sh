#!/usr/bin/env bash
# Start the judge and retrieval servers on a four-GPU node; stop both on exit.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN_DIR=${BROWSECOMP_PLUS_SERVICE_RUN_DIR:-${PROJECT_DIR}/runs/browsecomp_plus_services}
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-Qwen/Qwen3.8-27B}
JUDGE_MODEL_NAME=${BROWSECOMP_PLUS_JUDGE_MODEL:-Qwen/Qwen3.8-27B}
INDEX_GLOB=${BROWSECOMP_PLUS_INDEX_GLOB:-${PROJECT_DIR}/data/browsecomp_plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl}
JUDGE_PORT=${JUDGE_PORT:-9000}
RETRIEVER_PORT=${RETRIEVER_PORT:-8080}
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_DIR}"
# Node-local compilation caches avoid stale shared-filesystem handles.
CACHE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/recap-judge-XXXXXX")
CUDA_VISIBLE_DEVICES=${JUDGE_GPUS:-0} VLLM_CACHE_ROOT=${CACHE_DIR}/vllm \
    TORCHINDUCTOR_CACHE_DIR=${CACHE_DIR}/inductor TRITON_CACHE_DIR=${CACHE_DIR}/triton \
    vllm serve "${JUDGE_MODEL_PATH}" --served-model-name "${JUDGE_MODEL_NAME}" \
    --host 0.0.0.0 --port "${JUDGE_PORT}" --dtype bfloat16 --max-model-len 4096 \
    --max-num-seqs 8 --gpu-memory-utilization 0.90 --generation-config vllm \
    >"${RUN_DIR}/judge.log" 2>&1 &
judge_pid=$!
CUDA_VISIBLE_DEVICES=${RETRIEVER_GPUS:-1,2,3} python -m agent_system.envs.browsecomp_plus.retrieval_server \
    --index-path "${INDEX_GLOB}" --model-name Qwen/Qwen3-Embedding-8B \
    --torch-dtype bfloat16 --corpus-torch-dtype float32 --gpu-workers 3 \
    --max-length 512 --snippet-max-tokens 512 --document-max-tokens 4096 \
    --batch-size 16 --batch-timeout-ms 5 --k 5 --host 0.0.0.0 --port "${RETRIEVER_PORT}" \
    >"${RUN_DIR}/retriever.log" 2>&1 &
retriever_pid=$!
cleanup() {
    kill "${judge_pid}" "${retriever_pid}" 2>/dev/null || true
    wait "${judge_pid}" "${retriever_pid}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'Judge: http://<service-host>:%s; retriever: http://<service-host>:%s/mcp\n' "${JUDGE_PORT}" "${RETRIEVER_PORT}"
echo "Wait for both server logs to report readiness before starting training or evaluation."
wait -n "${judge_pid}" "${retriever_pid}"
