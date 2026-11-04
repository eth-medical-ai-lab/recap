"""Evaluate one HF model on SWE tasks, resuming only unfinished episodes.

Run with ``python -m scripts.swe_agent.eval.evaluate --help`` from the verl root.
GPU imports stay in the execution path so preparation and summaries work on CPU.
"""

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from uuid import uuid4


def write_json(path, value):
    """Replace a result atomically so interrupted writes remain resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent), version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{(root / 'verl/trainer/config').as_posix()}]",
                *overrides,
            ],
        )
    # PPO's unused reward/trainer sections may have mandatory placeholders.
    OmegaConf.to_container(config.evaluation, resolve=True, throw_on_missing=True)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    settings = config.evaluation
    for key in ("model_path", "output_dir", "data_path"):
        settings[key] = str(Path(settings[key]).expanduser().resolve())
    config.actor_rollout_ref.model.path = settings.model_path
    config.trainer.gym_env_kwargs.data_path = settings.data_path
    name = Path(settings.output_dir).name
    config.trainer.experiment_name = name
    config.trainer.gym_env_kwargs.e2b_run_id = name
    config.actor_rollout_ref.rollout.trace.experiment_name = name
    config.trainer.gym_compaction_enabled = settings.max_compactions > 0
    return config


def sampling_params(config):
    """Use an explicit allowlist: no action or summary request may carry a seed."""
    sampling = config.actor_rollout_ref.rollout.val_kwargs
    if config.actor_rollout_ref.rollout.full_determinism or "seed" in sampling:
        raise ValueError("Evaluation requires full_determinism=false and no request seed.")
    return dict(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        repetition_penalty=1.0,
        logprobs=False,
    )


def prepare_run(config):
    """Freeze settings and ordered task identities before any GPU or sandbox work."""
    from omegaconf import OmegaConf

    from agent_system.envs.swe_agent.env import SweAgentEnv

    settings = config.evaluation
    sampling_params(config)
    for key in ("samples_per_task", "max_active_episodes", "workers", "nodes", "gpus_per_node"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer.")
    if settings.max_tasks is not None and (type(settings.max_tasks) is not int or settings.max_tasks <= 0):
        raise ValueError("evaluation.max_tasks must be null or a positive integer.")
    if settings.max_compactions < 0:
        raise ValueError("evaluation.max_compactions cannot be negative.")
    model = Path(settings.model_path)
    if not (model / "config.json").is_file():
        raise ValueError("model_path must be a Hugging Face model directory; export raw checkpoints first.")
    specs = SweAgentEnv.list_task_specs(
        split=settings.split,
        dataset=settings.dataset,
        data_path=settings.data_path,
    )
    if settings.max_tasks is not None:
        specs = specs[: settings.max_tasks]
    rows = [
        dict(
            data_source=settings.dataset,
            agent_name="gym_agent",
            prompt=[dict(role="user", content="SWE task placeholder. SweAgentEnv builds the pinned task prompt.")],
            ability="swe_agent",
            reward_model=dict(style="rule", ground_truth=task["task_id"]),
            extra_info=dict(
                env_name="swe_agent",
                split=settings.split,
                task_id=task["task_id"],
                image_name=task["image_name"],
                template_name=task["template_name"],
                repo=task["repo"],
                index=i,
            ),
            uid=f"swe_agent:{settings.split}:{task['task_id']}",
        )
        for i, task in enumerate(specs)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    run = Path(settings.output_dir)
    saved_config = OmegaConf.to_container(config, resolve=True)
    protocol = dict(
        tasks=len(rows),
        samples_per_task=settings.samples_per_task,
        total_episodes=len(rows) * settings.samples_per_task,
        model=settings.model_path,
        model_step=settings.model_step,
        dataset=settings.dataset,
        split=settings.split,
        sampling=sampling_params(config),
        sampling_seed_policy="no_per_request_seed",
    )
    if settings.dataset == "swebench_verified":
        protocol.update(git_checkout="single_base_commit")

    # A different model, task subset, or decoding protocol must use a new directory.
    if (run / "config.json").exists():
        if json.loads((run / "config.json").read_text()) != saved_config:
            raise ValueError("Saved configuration differs; use a new output_dir for changed settings.")
        if json.loads((run / "protocol.json").read_text()) != protocol or (run / "rows.jsonl").read_bytes() != payload:
            raise ValueError("Saved task/protocol data differs; refusing to mix evaluations.")
    else:
        if (run / "results").exists():
            raise ValueError("Results exist without this evaluator's frozen configuration.")
        write_json(run / "protocol.json", protocol)
        (run / "rows.jsonl").write_bytes(payload)
        write_json(run / "config.json", saved_config)
    return rows


def pending_items(rows, run, samples):
    """Completed failures count as results; only missing/invalid episodes are retried."""
    pending = []
    for sample in range(samples):
        for index, row in enumerate(rows):
            path = run / "results" / f"{index:03d}_{sample}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved["task_id"], saved["sample_index"], saved["rollout_index"]) != (
                    row["extra_info"]["task_id"],
                    index,
                    sample,
                ):
                    raise ValueError(f"Result identity mismatch: {path}")
                if saved["status"] == "completed":
                    assert saved["reward"] in (0, 1)
                    continue
            pending.append((index, sample, row))
    return pending


def episode_result(output):
    if isinstance(output, list):
        assert len(output) == 1, "Validation must return one task-level result."
        output = output[0]
    fields = output.extra_fields
    invalid = bool(fields.get("swe_infrastructure_failure")) or fields.get("finish_reason") in (
        "rollout_timeout",
        "environment_error",
    )
    if not invalid:
        assert output.reward_score in (0, 1)
    return dict(
        status="environment_failure" if invalid else "completed",
        reward=None if invalid else output.reward_score,
        metadata=fields,
        num_turns=output.num_turns,
        metrics=output.metrics.model_dump(),
    )


def summarize(rows, run, samples):
    """Report valid-only accuracy/tool calls, with unresolved coverage made explicit."""
    pending = pending_items(rows, run, samples)
    completed = []
    for sample in range(samples):
        for index in range(len(rows)):
            path = run / "results" / f"{index:03d}_{sample}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["status"] == "completed":
                    completed.append(record)
    successes = sum(record["reward"] for record in completed)
    summary = dict(
        tasks=len(rows),
        samples_per_task=samples,
        expected_rollouts=samples * len(rows),
        completed=len(completed),
        pending=len(pending),
        successes=successes,
        complete=not pending,
        denominator="valid_completed_rollouts",
        **{f"mean_at_{samples}": None},
        avg_tool_calls=None,
    )
    if completed:
        summary[f"mean_at_{samples}"] = successes / len(completed)
        summary["avg_tool_calls"] = sum(r["metadata"]["episode_tool_call_count"] for r in completed) / len(completed)
    write_json(run / "summary.json", summary)
    return summary


class EvaluationWorker:
    """Keep a fixed number of whole episodes active, including sandbox setup and grading."""

    def __init__(self, config, client, attempt_id, slots):
        from omegaconf import OmegaConf

        from agent_system.gym_agent_loop import GymAgentLoop
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap, ToolListWrap

        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        config.trainer.gym_env_kwargs.e2b_create_concurrency = slots
        worker = AgentLoopWorker(config, client)
        self.config = config
        self.run_dir = Path(config.evaluation.output_dir)
        self.attempt_id = attempt_id
        self.slots = slots
        self.loop_class = GymAgentLoop
        self.loop_kwargs = dict(
            trainer_config=DictConfigWrap(config),
            server_manager=worker.llm_client,
            tokenizer=worker.tokenizer,
            processor=worker.processor,
            dataset_cls=worker.dataset_cls,
            data_config=DictConfigWrap(config.data),
            tools=ToolListWrap(worker.tools),
        )

    async def run(self, items):
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        async def consume():
            while not queue.empty():
                item = queue.get_nowait()
                try:
                    await self.evaluate(item)
                finally:
                    queue.task_done()

        tasks = [asyncio.create_task(consume()) for _ in range(self.slots)]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Finish cancellation/result writes before the driver makes its summary.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluate(self, item):
        index, sample, row = item
        task_id = row["extra_info"]["task_id"]
        started = time.monotonic()
        result = dict(
            task_id=task_id,
            sample_index=index,
            rollout_index=sample,
            status="interrupted",
            reward=None,
            attempt_id=self.attempt_id,
        )
        try:
            loop = self.loop_class(**self.loop_kwargs)
            output = await loop.run(
                sampling_params(self.config),
                raw_prompt=row["prompt"],
                extra_info=row["extra_info"],
                env_name="swe_agent",
                task_id=task_id,
                split=self.config.evaluation.split,
                validate=True,
                global_steps=self.config.evaluation.model_step,
                sample_index=index,
                rollout_n=sample,
                log_trajectory_txt=True,
            )
            result.update(episode_result(output))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result.update(
                status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
            )
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(self.run_dir / "results" / f"{index:03d}_{sample}.json", result)
            print(f"RESULT {index:03d}/{sample} {result['status']} reward={result['reward']}", flush=True)


async def evaluate(config, items, attempt_id):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Initialize verl's agent registry before the server manager imports it.
    from verl.experimental.agent_loop import agent_loop  # noqa: F401
    from verl.workers.rollout.llm_server import LLMServerManager

    settings = config.evaluation
    nodes = sorted(
        n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) >= settings.gpus_per_node
    )
    if len(nodes) != settings.nodes:
        raise RuntimeError(f"Expected {settings.nodes} GPU nodes, found {len(nodes)}.")
    servers = await LLMServerManager.create(config=config)
    count = min(settings.workers, settings.max_active_episodes, len(items))
    slots, remainder = divmod(settings.max_active_episodes, count)
    worker_class = ray.remote(num_cpus=0.5, max_concurrency=1)(EvaluationWorker)
    workers = [
        worker_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[i % len(nodes)], soft=False)
        ).remote(config, servers.get_client(), attempt_id, slots + (i < remainder))
        for i in range(count)
    ]
    calls = [worker.run.remote(items[i::count]) for i, worker in enumerate(workers)]
    # Wait for all workers before summarizing, even if one encounters a fatal error.
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Freeze inputs without Ray, GPUs, or sandboxes.")
    modes.add_argument("--summarize-only", action="store_true", help="Recompute summary.json without running episodes.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. evaluation.model_path=/path/to/HF/model")
    args = parser.parse_args()
    config = load_config(args.overrides)
    run = Path(config.evaluation.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    # Linux: prevent two writers from evaluating the same output directory.
    import fcntl

    with (run / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare_run(config)
        if args.prepare_only:
            print(f"PREPARED {run}: {len(rows)} tasks", flush=True)
            return
        samples = config.evaluation.samples_per_task
        items = pending_items(rows, run, samples)
        if args.summarize_only or not items:
            print(json.dumps(summarize(rows, run, samples)), flush=True)
            return

        attempt_id = os.environ.get("SLURM_JOB_ID") or uuid4().hex
        attempt = run / "attempts" / attempt_id
        for index, sample, _ in items:
            previous = run / "results" / f"{index:03d}_{sample}.json"
            if previous.exists():
                write_json(attempt / "previous_errors" / previous.name, json.loads(previous.read_text()))
        write_json(
            attempt / "work_items.json",
            dict(
                pending=len(items),
                items=[dict(index=i, sample=s, task_id=r["extra_info"]["task_id"]) for i, s, r in items],
            ),
        )
        for key, folder in [
            ("gym_trajectory_dir", "gym_trajectories"),
            ("gym_swebench_failure_dir", "failures"),
            ("gym_swebench_solution_dir", "solutions"),
        ]:
            config.trainer[key] = str(attempt / folder)
        # Match job-exit cleanup after validating the portable, frozen evaluation configuration.
        config.trainer.gym_env_kwargs.e2b_run_id = os.environ.get("SLURM_JOB_NAME", config.trainer.experiment_name)
        write_json(attempt / "effective_config.json", OmegaConf.to_container(config, resolve=True))
        print(f"WORK_ITEMS completed={samples * len(rows) - len(items)} pending={len(items)}", flush=True)
        import ray

        ray.init(address=os.environ.get("RAY_ADDRESS"))
        try:
            asyncio.run(evaluate(config, items, attempt_id))
        finally:
            ray.shutdown()
            print("EVALUATION_SUMMARY", json.dumps(summarize(rows, run, samples)), flush=True)


if __name__ == "__main__":
    main()
