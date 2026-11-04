#!/usr/bin/env bash
# RECAP on browsecomp_plus: GRPO, LOO summary baseline, SD-only signal clipping.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/browsecomp_plus}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-recap_browsecomp_plus}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/runs/${EXPERIMENT_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
TRAIN_NNODES=${TRAIN_NNODES:-2}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
LOGGER=${LOGGER:-'["console"]'}
RESUME_MODE=${RESUME_MODE:-disable}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
RECAP_COEF=${RECAP_COEF:-0.1}
SUMMARY_PG_COEF=${SUMMARY_PG_COEF:-null}
CONTINUATION_SD_COEF=${CONTINUATION_SD_COEF:-null}
GYM_COMPACTION_ENABLED=${GYM_COMPACTION_ENABLED:-True}
GYM_COMPACTION_MAX_COMPACTIONS=${GYM_COMPACTION_MAX_COMPACTIONS:-3}
COMPACTION_DISTORTION_ENABLED=${COMPACTION_DISTORTION_ENABLED:-True}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/rollouts/train.jsonl}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/rollouts/test.jsonl}
GYM_MAX_EPISODE_TOKENS=${GYM_MAX_EPISODE_TOKENS:-null}
BROWSECOMP_PLUS_MCP_URL=${BROWSECOMP_PLUS_MCP_URL:-http://127.0.0.1:8080/mcp}
BROWSECOMP_PLUS_JUDGE_URL=${BROWSECOMP_PLUS_JUDGE_URL:-http://127.0.0.1:9000}
BROWSECOMP_PLUS_JUDGE_MODEL=${BROWSECOMP_PLUS_JUDGE_MODEL:-Qwen/Qwen3.8-27B}
TRAIN_TEMPERATURE=1.0
TRAIN_TOP_P=1.0
TRAIN_TOP_K=-1
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
VAL_TOP_K=20

# Keep task, summary and continuation samples in the same logical trajectory.
OVERRIDES=(
    hydra.run.dir="${RUN_DIR}/hydra/${SLURM_JOB_ID:-local}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PROJECT_DIR}"
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.filter_groups.enable=False
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=True
    algorithm.rollout_correction.loss_type=ppo_clip
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
    algorithm.compaction_distortion.enabled=${COMPACTION_DISTORTION_ENABLED}
    algorithm.compaction_distortion.coef=${RECAP_COEF}
    algorithm.compaction_distortion.summary_pg_coef=${SUMMARY_PG_COEF}
    algorithm.compaction_distortion.continuation_sd_coef=${CONTINUATION_SD_COEF}
    algorithm.compaction_distortion.summary_baseline=leave_one_out
    algorithm.compaction_distortion.scoring_policy=pi_rollout
    algorithm.compaction_distortion.sd_advantage_clip=5.0
    algorithm.compaction_distortion.clipping=joint
    algorithm.compaction_distortion.distortion_measure_interval=0
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=0
    data.gen_batch_size=1
    data.val_batch_size=150
    data.train_max_samples=-1
    data.val_max_samples=150
    data.max_prompt_length=16384
    data.max_response_length=131072
    data.return_raw_chat=True
    data.filter_overlong_prompts=False
    data.truncation=error
    data.shuffle=True
    data.validation_shuffle=False
    data.dataloader_num_workers=0
    data.seed=null
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.continuous_token.enable=True
    data.continuous_token.model_family=qwen35
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.clip_ratio=0.20
    actor_rollout_ref.actor.clip_ratio_low=0.20
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.loss_agg_mode=trajectory-mean-token-mean
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.agent.default_agent_loop=gym_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=agent_system/agent_loop.yaml
    actor_rollout_ref.rollout.agent.num_workers=48
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100
    actor_rollout_ref.rollout.prompt_length=16384
    actor_rollout_ref.rollout.response_length=131072
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.max_num_seqs=8
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40
    actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.70
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K}
    +actor_rollout_ref.rollout.repetition_penalty=1.0
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.full_determinism=False
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=4
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=recap_browsecomp_plus
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${TRAIN_NNODES}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir=null
    trainer.validation_data_dir=null
    trainer.log_train_trajectory_txt=1
    trainer.log_val_trajectory_txt=1
    trainer.gym_trajectory_dir="${RUN_DIR}/gym_trajectories"
    trainer.gym_max_tokens=8192
    trainer.gym_max_episode_tokens=${GYM_MAX_EPISODE_TOKENS}
    +trainer.gym_env_kwargs="{ray_pool_size: 1, ray_worker_num_cpus: 0.5, mcp_url: '${BROWSECOMP_PLUS_MCP_URL}', judge_url: '${BROWSECOMP_PLUS_JUDGE_URL}', judge_model: '${BROWSECOMP_PLUS_JUDGE_MODEL}'}"
    ++trainer.gym_compaction_enabled=${GYM_COMPACTION_ENABLED}
    ++trainer.gym_compaction_max_compactions=${GYM_COMPACTION_MAX_COMPACTIONS}
    ++trainer.gym_compaction_keep_recent_turns=1
    ++trainer.gym_compaction_max_tokens=8192
    ++trainer.gym_compaction_context_threshold=32768
    trainer.save_freq=10
    trainer.test_freq=20
    trainer.max_actor_ckpt_to_keep=2
    trainer.total_epochs=60
    trainer.total_training_steps=150
    trainer.val_before_train=True
    trainer.resume_mode=${RESUME_MODE}
    trainer.resume_from_path=null
    rollout.nnodes=${ROLLOUT_NNODES}
    rollout.n_gpus_per_node=${GPUS_PER_NODE}
    rollout.n=8
    rollout.total_rollout_steps=2400
    async_training.require_batches=2
    async_training.trigger_parameter_sync_step=1
    async_training.staleness_threshold=0.5
    async_training.partial_rollout=True
    async_training.use_trainer_do_validate=False
    async_training.use_dynamic_resource_scheduling=True
    async_training.dynamic_schedule_policy=default
    async_training.dynamic_schedule_deactivate_ratio=0.3
    async_training.dynamic_schedule_enable_rebalance=True
    async_training.concurrent_samples_per_replica=8
)

# Config inspection is offline and never reads or prints credentials.
if [[ "${PRINT_CONFIG_ONLY:-0}" == "1" ]]; then
    printf '%s\n' "${OVERRIDES[@]}" "$@"
    exit 0
fi
for path in "${TRAIN_FILE}" "${TEST_FILE}"; do
    test -s "${path}" || { echo "Missing prepared data: ${path}" >&2; exit 1; }
done
if [[ "${RESUME_MODE}" == "disable" && -d "${CHECKPOINT_DIR}" ]] &&
    [[ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Choose a new CHECKPOINT_DIR or an explicit resume mode; checkpoint directory is not empty." >&2
    exit 1
fi
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=true VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GYM_TRAJECTORY_DIR=${RUN_DIR}/gym_trajectories
cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${GYM_TRAJECTORY_DIR}"
printf '%s\n' "${OVERRIDES[@]}" "$@" >"${RUN_DIR}/training_overrides_${SLURM_JOB_ID:-local}.txt"
exec python -m verl.experimental.fully_async_policy.fully_async_main "${OVERRIDES[@]}" "$@"
