# BrowseComp-Plus training and evaluation

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

Prepare the data and retrieval index:

```bash
python -m scripts.browsecomp_plus.train.prepare_browsecomp_plus_data --output-dir data/browsecomp_plus
python -m scripts.browsecomp_plus.train.prepare_retrieval
```

Preparation writes the final 300 training questions and 150 held-out test questions
to `data/browsecomp_plus/tasks/{train,test}.jsonl`, with matching inputs under
`rollouts/{train,test}.jsonl`. The two splits do not overlap.

Start the retrieval and judge services on a separate four-GPU node:

```bash
bash scripts/browsecomp_plus/train/serve_browsecomp_plus_services.sh
```

## Training

The default configuration uses **16 GPUs**: `TRAIN_NNODES=2` for training,
`ROLLOUT_NNODES=2` for rollout, and `GPUS_PER_NODE=4` for both. These resources
exclude the retrieval and judge services. Set these variables to match your cluster.

Start or connect to a [Ray cluster](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)
with these resources. Use the same Python environment, repository path, and data
paths on every node, and include the repository root in `PYTHONPATH` before
starting Ray on each node.

Once Ray and both services are ready, launch Qwen3.5-4B training from the Ray head
node. Replace `HEAD_IP:6379` with your Ray cluster address:

```bash
export BROWSECOMP_PLUS_MCP_URL=http://SERVICE_HOST:8080/mcp
export BROWSECOMP_PLUS_JUDGE_URL=http://SERVICE_HOST:9000
export MODEL_PATH=/path/to/Qwen3.5-4B
export EXPERIMENT_NAME=recap_browsecomp_plus
export RAY_ADDRESS=HEAD_IP:6379
bash scripts/browsecomp_plus/train/train.sh
```

Training settings are in [train.sh](train/train.sh). Set `RECAP_COEF=0` for
task-reward-only GRPO with context compaction.

Summary PG uses the raw mean teacher–student log probability difference. Continuation SD clips
each token difference to ±5 before dividing by continuation length. Pass
`algorithm.compaction_distortion.sd_advantage_clip=null` to disable this clip.
To enable the FSDP2 EMA teacher, append `algorithm.compaction_distortion.teacher_ema_rate=0.05`
to the training command. Resume with the same rate and actor world size.

## Evaluation

Use a Qwen3.5-4B checkpoint exported to Hugging Face format. Evaluation uses the
150 held-out questions prepared above and defaults to **4 GPUs** on one node,
in addition to the retrieval/judge services. Keep both service URLs exported.
The evaluator starts Ray locally if needed.

```bash
python -m scripts.browsecomp_plus.eval.evaluate \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/bc_eval
python -m scripts.browsecomp_plus.eval.evaluate --summarize-only \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/bc_eval
```

Settings are in [eval/config.yaml](eval/config.yaml). For full-context evaluation,
add `evaluation.max_compactions=0` to both commands. Repeat the same command and
output directory to finish pending episodes.

For the native tool-calling baseline, use:

```bash
python -m scripts.browsecomp_plus.eval.evaluate \
  evaluation.model_path=/path/to/Qwen3.5-4B evaluation.output_dir=runs/bc_native \
  evaluation.backend=native evaluation.max_compactions=0
```

Summarize with the same overrides and `--summarize-only`.

## Slurm (optional)

Alternatively, activate the runtime and export the service/model settings above,
then submit from the repository root:

```bash
sbatch scripts/browsecomp_plus/train/train.slurm
```

The launcher starts Ray automatically and requests four nodes with four GPUs each.
If you change the allocation in [train.slurm](train/train.slurm), match it with
`TRAIN_NNODES`, `ROLLOUT_NNODES`, and `GPUS_PER_NODE`.

For evaluation, the same overrides can be passed to:

```bash
sbatch scripts/browsecomp_plus/eval/evaluate.slurm \
  evaluation.model_path=/path/to/model evaluation.output_dir=runs/bc_eval
```
