# SWE-agent training and evaluation with E2B

Run all commands from the repository root on Linux. Use Python 3.12, CUDA 13,
PyTorch 2.11.0, vLLM 0.26.0 and Transformers 5.5.3 with verl v0.9.0's FSDP2 runtime.
Install the additional dependencies and Qwen3.5 kernels:

```bash
python -m pip install -r requirements.txt
MAX_JOBS=8 python -m pip install --no-build-isolation --no-deps \
  flash-attn==2.8.3 causal-conv1d==1.6.2.post1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Data and environment

Download the final 300 SWE-Smith training tasks and all 500 SWE-bench Verified
test tasks from the pinned public datasets, then build the E2B sandbox templates.
The training task IDs are fixed in [train_300_selection.json](train/train_300_selection.json).

```bash
export E2B_API_KEY=YOUR_E2B_API_KEY
export DATA_DIR="$PWD/data/swe_agent"
python -m scripts.swe_agent.train.prepare_swe_training_data \
  --output-dir "$DATA_DIR"
python -m scripts.swe_agent.train.prepare_swebench_verified_data \
  --output-dir "$DATA_DIR"
python -m scripts.swe_agent.train.build_e2b_templates \
  "$DATA_DIR/rollouts/train.jsonl" "$DATA_DIR/rollouts/test.jsonl" \
  --ledger "$DATA_DIR/e2b_templates.json"
```

Task records are saved under `data/swe_agent/tasks/{train,test}.jsonl`, with
training/evaluation inputs under `rollouts/{train,test}.jsonl`. The training input
contains eight fixed shuffled passes over the 300 tasks (2,400 rows). Raw data is
downloaded locally; the repository contains only the selection and preparation code.

## Training

The default configuration uses **16 GPUs**: `TRAIN_NNODES=2` for training,
`ROLLOUT_NNODES=2` for rollout, and `GPUS_PER_NODE=4` for both. Set these variables
to match your cluster.

Start or connect to a [Ray cluster](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)
with these resources. Use the same Python environment, repository path, and data
paths on every node. Before starting Ray on each node, include the repository root
in `PYTHONPATH` and export `E2B_API_KEY`.

Launch Qwen3.5-4B training from the Ray head node. Replace `HEAD_IP:6379` with your
Ray cluster address:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export EXPERIMENT_NAME=recap_swe
export RAY_ADDRESS=HEAD_IP:6379
bash scripts/swe_agent/train/train.sh
```

Training settings are in [train.sh](train/train.sh). Adjust `ROLLOUT_AGENT_WORKERS` and
`E2B_CREATE_CONCURRENCY` to your E2B capacity. Set `RECAP_COEF=0` for
task-reward-only GRPO with context compaction.

Summary PG uses the raw mean teacher–student log probability difference. Continuation SD clips
each token difference to ±5 before dividing by continuation length. Pass
`algorithm.compaction_distortion.sd_advantage_clip=null` to disable this clip.
To enable the FSDP2 EMA teacher, append `algorithm.compaction_distortion.teacher_ema_rate=0.05`
to the training command. Resume with the same rate and actor world size.

## Evaluation

Use a Qwen3.5-4B checkpoint exported to Hugging Face format. Evaluation uses all
500 Verified tasks prepared above and defaults to **4 GPUs** on one node.
Keep `E2B_API_KEY` exported. The evaluator starts Ray locally if needed.

```bash
python -m scripts.swe_agent.eval.evaluate \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/swe_eval
python -m scripts.swe_agent.eval.evaluate --summarize-only \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/swe_eval
```

Settings are in [eval/config.yaml](eval/config.yaml). For full-context evaluation,
add `evaluation.max_compactions=0 evaluation.max_model_len=131072` to both commands.
Repeat the same command and output directory to finish pending episodes.

### Native tool-calling baseline

Copy [native_example.json](eval/native_example.json) and set the checkpoint path,
revision, data path and output directory. The default uses one node with four GPUs.
Run preparation, evaluation and summarization in order:

```bash
python -m scripts.swe_agent.eval.evaluate_native /path/to/config.json --prepare
python -m scripts.swe_agent.eval.evaluate_native /path/to/config.json
python -m scripts.swe_agent.eval.evaluate_native /path/to/config.json --summarize
```

Episodes are attempted once per run. Infrastructure failures remain pending for
the next run; completed model failures score zero. Internal HTTP, sandbox creation
and grading retries remain enabled.

## Slurm (optional)

Alternatively, activate the runtime and export the E2B/model settings above,
then submit from the repository root:

```bash
sbatch scripts/swe_agent/train/train.slurm
```

The launcher starts Ray automatically and requests four nodes with four GPUs each.
If you change the allocation in [train.slurm](train/train.slurm), match it with
`TRAIN_NNODES`, `ROLLOUT_NNODES`, and `GPUS_PER_NODE`.

For evaluation:

```bash
sbatch scripts/swe_agent/eval/evaluate.slurm \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/swe_eval
sbatch scripts/swe_agent/eval/evaluate_native.slurm /path/to/config.json
```

Native evaluation runs preparation and summarization once, and one evaluation
process per node. For multiple nodes, set `nodes` in the JSON and request the same
number with `sbatch --nodes=N`; keep `gpus_per_node` consistent with the allocation.
