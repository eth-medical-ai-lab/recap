"""BrowseComp-Plus evaluation for Qwen3.5-4B, full context or compaction.

Run from the verl root with ``python -m scripts.browsecomp_plus.eval.evaluate``.
Preparation and summaries do not import Ray, torch or the model runtime.
"""

import argparse
import asyncio
import json
import os
import secrets
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .records import pending_items, read_record, summarize, write_json


def load_config(overrides):
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent), version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{(root / 'verl/trainer/config').as_posix()}]",
                *overrides,
            ],
        )
    OmegaConf.to_container(config.evaluation, resolve=True, throw_on_missing=True)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    settings = config.evaluation
    for key in ("model_path", "output_dir", "data_path", "task_path"):
        settings[key] = str(Path(settings[key]).expanduser().resolve())
    run = Path(settings.output_dir)
    config.actor_rollout_ref.model.path = settings.model_path
    config.actor_rollout_ref.rollout.trace.experiment_name = run.name
    config.actor_rollout_ref.rollout.prometheus.served_model_name = settings.model_path
    config.trainer.experiment_name = run.name
    config.trainer.gym_trajectory_dir = str(run / "gym_trajectories")
    config.trainer.gym_env_kwargs.data_path = str(run / "tasks.jsonl")
    config.trainer.gym_compaction_enabled = settings.max_compactions > 0
    return config


def sampling_params(config):
    """An allowlist keeps fixed seeds out of both action and summary requests."""
    rollout = config.actor_rollout_ref.rollout
    val = rollout.val_kwargs
    if rollout.full_determinism or "seed" in val:
        raise ValueError("Use full_determinism=false and no request-level seed.")
    return dict(temperature=val.temperature, top_p=val.top_p, top_k=val.top_k, repetition_penalty=1.0, logprobs=False)


def initialize_sampling(config):
    """vLLM needs an integer engine seed; use new OS entropy on every startup."""
    sampling_params(config)
    rollout = config.actor_rollout_ref.rollout
    if rollout.seed is not None or "seed" in (rollout.engine_kwargs.get("vllm") or {}):
        raise ValueError("Fixed engine seeds are disabled for this evaluator.")
    rollout.seed = secrets.randbelow(2**31 - 4)


def prepare_run(config):
    """Freeze the task set and protocol so retries cannot mix different evaluations."""
    settings = config.evaluation
    sampling_params(config)
    if settings.backend not in ("gym", "native"):
        raise ValueError("evaluation.backend must be gym or native.")
    if settings.backend == "native" and settings.max_compactions != 0:
        raise ValueError("Native evaluation requires evaluation.max_compactions=0.")
    for key in (
        "samples_per_task",
        "workers",
        "judge_workers",
        "max_model_len",
        "max_turns",
        "max_response_tokens",
        "max_summary_tokens",
    ):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer.")
    if settings.max_tasks is not None and (type(settings.max_tasks) is not int or settings.max_tasks <= 0):
        raise ValueError("max_tasks must be null or a positive integer.")
    if settings.max_compactions < 0 or not 0 < settings.compaction_threshold < settings.max_model_len:
        raise ValueError("Invalid compaction count or threshold.")
    rollout = config.actor_rollout_ref.rollout
    if rollout.seed is not None or "seed" in (rollout.engine_kwargs.get("vllm") or {}):
        raise ValueError("Do not configure a fixed engine seed.")
    model = Path(settings.model_path)
    if not (model / "config.json").is_file():
        raise ValueError("model_path must be an HF model directory; export raw checkpoints first.")
    payload = Path(settings.data_path).read_bytes()
    task_payload = Path(settings.task_path).read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    rows = rows[: settings.max_tasks]
    ids = [str(r["extra_info"]["task_id"]) for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation task IDs must be nonempty and unique.")
    tasks = [json.loads(line) for line in task_payload.decode("utf-8").splitlines() if line.strip()]
    by_id = {str(t["query_id"]): t for t in tasks}
    assert len(by_id) == len(tasks)
    selected = [by_id[q] for q in ids]
    assert all(t["query"] and t["answer"] for t in selected)
    run = Path(settings.output_dir)
    protocol = dict(
        questions=len(rows),
        samples_per_question=settings.samples_per_task,
        model_path=settings.model_path,
        checkpoint_step=settings.model_step,
        sampling=sampling_params(config),
        sampling_seed_policy="fresh_os_entropy_engine_no_request_seed",
        compaction_threshold=settings.compaction_threshold,
        max_compactions=settings.max_compactions,
        judge_model=config.trainer.gym_env_kwargs.judge_model,
    )
    if settings.backend == "native":
        protocol.update(
            backend="native",
            native=dict(
                tool_parser="qwen3_coder",
                chat_template_kwargs=OmegaConf.to_container(config.data.apply_chat_template_kwargs, resolve=True),
                language_model_only=True,
            ),
        )
    saved_config = OmegaConf.to_container(config, resolve=True)
    rows_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tasks_text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in selected)
    if (run / "config.json").exists():
        if (
            json.loads((run / "config.json").read_text()) != saved_config
            or json.loads((run / "protocol.json").read_text()) != protocol
            or (run / "rows.jsonl").read_text(encoding="utf-8") != rows_text
            or (run / "tasks.jsonl").read_text(encoding="utf-8") != tasks_text
        ):
            raise ValueError("Saved inputs differ; use a new output_dir for changed settings.")
    else:
        if (run / "rollouts").exists() or (run / "judgments").exists():
            raise ValueError("Existing results do not belong to this evaluator; use a new output_dir.")
        run.mkdir(parents=True, exist_ok=True)
        (run / "rows.jsonl").write_text(rows_text, encoding="utf-8")
        (run / "tasks.jsonl").write_text(tasks_text, encoding="utf-8")
        write_json(run / "protocol.json", protocol)
        write_json(run / "config.json", saved_config)
    return rows


def episode_result(output):
    """Preserve environment outcomes, deferring all answer grading until collection ends."""
    if isinstance(output, list):
        assert len(output) == 1
        output = output[0]
    fields = output.extra_fields
    final = next((info for info in reversed(fields.get("env_step_infos", [])) if "final_answer" in info), {})
    answer = final.get("final_answer")
    assert bool(fields.get("episode_answer_submitted")) == (answer is not None)
    return dict(
        status="error" if fields["finish_reason"] in ("rollout_timeout", "environment_error") else "completed",
        final_answer=answer,
        finish_reason=fields["finish_reason"],
        tool_calls=fields["episode_tool_call_count"],
        metadata={k: v for k, v in fields.items() if k not in ("env_step_infos", "final_observation_messages")},
    )


class EvaluationWorker:
    """Run one episode at a time; worker count is the whole-episode concurrency cap."""

    def __init__(self, config, client, attempt):
        from agent_system.gym_agent_loop import GymAgentLoop
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap, ToolListWrap

        worker = AgentLoopWorker(config, client)
        self.config, self.attempt = config, attempt
        self.run_dir = Path(config.evaluation.output_dir)
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
        for index, sample, row in items:
            name = f"{index:03d}_{sample}.json"
            started = time.monotonic()
            result = dict(
                index=index,
                sample=sample,
                task_id=str(row["extra_info"]["task_id"]),
                status="interrupted",
                job_id=self.attempt,
                sampling=sampling_params(self.config),
            )
            try:
                loop = self.loop_class(**self.loop_kwargs)
                output = await loop.run(
                    sampling_params(self.config),
                    raw_prompt=row["prompt"],
                    extra_info=row["extra_info"],
                    env_name=row["extra_info"]["env_name"],
                    task_id=result["task_id"],
                    split=row["extra_info"]["split"],
                    validate=True,
                    env_kwargs={"data_path": self.config.trainer.gym_env_kwargs.data_path},
                    global_steps=self.config.evaluation.model_step,
                    sample_index=index,
                    rollout_n=sample,
                    log_trajectory_txt=True,
                )
                result.update(episode_result(output))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                result.update(status="error", error=str(error), traceback=traceback.format_exc())
            finally:
                result["elapsed_seconds"] = time.monotonic() - started
                write_json(self.run_dir / "attempts" / self.attempt / name, result)
                write_json(self.run_dir / "rollouts" / name, result)
                print(f"ROLLOUT {name} {result['status']} seconds={result['elapsed_seconds']:.1f}", flush=True)


async def collect(config, items, attempt):
    import ray

    from verl.experimental.agent_loop import agent_loop  # noqa: F401
    from verl.workers.rollout.llm_server import LLMServerManager

    servers = await LLMServerManager.create(config=config)
    assert len(servers.server_handles) == 4, "Use one rollout node with four TP1 policy replicas."
    count = min(config.evaluation.workers, len(items))
    worker_class = ray.remote(num_cpus=0.5, max_concurrency=1)(EvaluationWorker)
    workers = [worker_class.remote(config, servers.get_client(), attempt) for _ in range(count)]
    outcomes = await asyncio.gather(
        *(w.run.remote(items[i::count]) for i, w in enumerate(workers)), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


def score(row, task, judge_url, model):
    from agent_system.envs.browsecomp_plus.judge import exact_match, judge_answer

    row = dict(row)
    if row["final_answer"] is None:
        row.update(reward=0, reward_source="no_submitted_answer", judge=None)
    elif exact_match(task["answer"], row["final_answer"]):
        row.update(reward=1, reward_source="normalized_exact_match", judge=None)
    else:
        result = judge_answer(task["query"], task["answer"], row["final_answer"], judge_url=judge_url, model=model)
        row.update(reward=result.score, reward_source="llm_judge", judge=asdict(result))
    return row


def judge_and_summarize(config, rows, judge_url=None):
    """Resume unjudged rollouts; transport exceptions stay pending, not incorrect."""
    run = Path(config.evaluation.output_dir)
    task_lines = (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    tasks = {str(t["query_id"]): t for t in map(json.loads, task_lines)}
    completed, pending = [], []
    for sample in range(config.evaluation.samples_per_task):
        for index, row in enumerate(rows):
            task_id = row["extra_info"]["task_id"]
            path = run / "rollouts" / f"{index:03d}_{sample}.json"
            if not path.exists():
                continue
            record = read_record(path, index, sample, task_id)
            if record["status"] != "completed":
                continue
            target = run / "judgments" / path.name
            if target.exists():
                saved = read_record(target, index, sample, task_id)
                if saved["final_answer"] != record["final_answer"]:
                    raise ValueError(f"Judgment does not match the saved answer: {target}")
                completed.append(saved)
            else:
                pending.append((target, record))
    errors = []
    if judge_url is not None:
        with ThreadPoolExecutor(max_workers=config.evaluation.judge_workers) as pool:
            futures = {
                pool.submit(
                    score, record, tasks[record["task_id"]], judge_url, config.trainer.gym_env_kwargs.judge_model
                ): target
                for target, record in pending
            }
            for future in as_completed(futures):
                target = futures[future]
                try:
                    result = future.result()
                    write_json(target, result)
                    completed.append(result)
                except Exception as error:
                    errors.append(dict(sample=target.stem, error=str(error)))
    report = summarize(
        completed,
        [r["extra_info"]["task_id"] for r in rows],
        config.evaluation.samples_per_task,
        config.evaluation.max_summary_tokens,
    )
    if errors:
        report["judge_errors"] = errors
    write_json(run / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    modes.add_argument("--judge-only", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. evaluation.samples_per_task=4")
    args = parser.parse_args()
    config = load_config(args.overrides)
    run = Path(config.evaluation.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    # Linux execution: a second allocation must not write the same logical samples.
    import fcntl

    with (run / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare_run(config)
        if args.prepare_only:
            print(f"PREPARED {run}: {len(rows)} questions", flush=True)
            return
        if args.summarize_only:
            print(json.dumps(judge_and_summarize(config, rows), indent=2), flush=True)
            return
        items = pending_items(rows, run, config.evaluation.samples_per_task)
        if items and not args.judge_only:
            attempt = os.environ.get("SLURM_JOB_ID") or uuid4().hex
            initialize_sampling(config)
            config.trainer.gym_trajectory_dir = str(run / "gym_trajectories" / attempt)
            write_json(run / "attempts" / attempt / "actual_config.json", OmegaConf.to_container(config, resolve=True))
            if config.evaluation.backend == "native":
                from .native import collect_native

                asyncio.run(collect_native(config, items, attempt))
            else:
                import ray

                ray.init(address=os.environ.get("RAY_ADDRESS"))
                try:
                    asyncio.run(collect(config, items, attempt))
                finally:
                    ray.shutdown()
        report = judge_and_summarize(config, rows, config.trainer.gym_env_kwargs.judge_url)
        print(json.dumps(report, indent=2), flush=True)
        if not report["complete"]:
            raise RuntimeError("Evaluation incomplete; rerun with the same settings to resume missing results.")
        if config.evaluation.wandb:
            import wandb

            with wandb.init(
                project="browsecomp_plus",
                id=run.name,
                name=run.name,
                resume="allow",
                dir=str(run),
                config=json.loads((run / "protocol.json").read_text()),
            ) as tracked:
                tracked.log(
                    {"eval/" + k: v for k, v in report.items() if isinstance(v, int | float)},
                    step=config.evaluation.model_step,
                )
                tracked.summary.update(report)


if __name__ == "__main__":
    main()
