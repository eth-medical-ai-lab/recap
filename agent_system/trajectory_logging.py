"""Segment transcripts, optional W&B uploads, and standard SWE-bench solution records."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


def trajectory_directory(trainer, validate: bool, enabled: bool) -> Path | None:
    """Respect the trainer's trajectory sampling flags and existing output paths."""
    if not (int(trainer.get("log_val_trajectory_txt", 0) or 0) > 0 if validate else enabled):
        return None
    directory = os.environ.get("GYM_TRAJECTORY_DIR") or trainer.get("gym_trajectory_dir")
    if directory:
        return Path(directory)
    directory = trainer.get("validation_data_dir" if validate else "rollout_data_dir")
    return Path(directory) / "gym_trajectories" if directory else None


def write_trajectory_txt(path: Path, metadata: dict, segments) -> list[Path]:
    """Save each segment in its own file, including terminal feedback and errors."""
    paths = []
    for index, segment in enumerate(segments):
        segment_path = path.with_suffix(".txt") if len(segments) == 1 else path / f"{index:03d}_{segment.kind}.txt"
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            **metadata,
            "segment_index": index,
            "segment_count": len(segments),
            "segment_kind": segment.kind,
            "prompt_token_count": len(segment.prompt_ids),
            "response_token_count": len(segment.response_ids),
            "generated_token_count": sum(segment.response_mask),
            "min_global_steps": segment.min_global_step,
            "max_global_steps": segment.max_global_step,
        }
        with segment_path.open("w", encoding="utf-8") as file:
            file.write("Gym Agent Trajectory Segment\n\n")
            file.write(json.dumps(header, ensure_ascii=False, indent=2, default=str) + "\n\n")
            # Print text verbatim, with tool calls and IDs alongside it.
            for turn, message in enumerate(segment.messages, start=1):
                file.write(f"===== Turn {turn:03d} | {message['role'].upper()} =====\n")
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, indent=2, default=str)
                file.write(content + "\n")
                details = {key: value for key, value in message.items() if key not in {"role", "content"} and value}
                if details:
                    file.write(json.dumps(details, ensure_ascii=False, indent=2, default=str) + "\n")
                file.write("\n")
            file.write("===== End | SUMMARY =====\n")
            file.write(f"total_messages: {len(segment.messages)}\n")
            counts = Counter(message["role"] for message in segment.messages)
            file.write(f"role_turn_counts: {json.dumps(counts, sort_keys=True)}\n")
        paths.append(segment_path)
    return paths


def write_swebench_solution(path: Path, solution: dict[str, str]) -> None:
    """Save the solution patch with the keys required by the SWE-bench evaluator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(solution, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_swebench_failure(path: Path, metadata: dict) -> None:
    """Record a failed attempt separately from an evaluated solution."""
    failure = metadata["failure"]
    record = {
        "instance_id": str(metadata["task_id"]),
        "task_id": str(metadata["task_id"]),
        "split": metadata["split"],
        "failure_kind": failure["kind"],
        "message": failure["message"],
        "elapsed_seconds": metadata["rollout_elapsed_seconds"],
    }
    if metadata["rollout_timed_out"]:
        record["timeout_seconds"] = metadata["rollout_timeout_seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def log_trajectories_to_wandb(paths, scores, trainer, step: int, *, validate: bool) -> None:
    """Upload saved episodes, keeping all segments together and one table row per path."""
    # 1. Use the existing training/validation limits and active W&B run.
    split = "val" if validate else "train"
    trajectories_to_log = int(trainer.get(f"log_{split}_trajectory_txt", 0) or 0)
    if trajectories_to_log <= 0:
        return
    loggers = trainer.get("logger", [])
    if isinstance(loggers, str):
        loggers = [loggers]
    if "wandb" not in loggers:
        return

    # 2. Keep the first N distinct saved episodes, in batch order.
    table_key = f"{split}/trajectory_txt"
    samples = []
    seen = set()
    for path, score in zip(paths, scores, strict=False):
        if hasattr(path, "item"):
            path = path.item()
        if path is None:
            continue
        path = os.fspath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        samples.append((path, score))

    if not samples:
        print(f"Warning: {table_key} requested, but no gym trajectory txt files were found.")
        return
    samples = samples[:trajectories_to_log]

    # 3. Upload complete files with a bounded preview; logging failures must not stop training.
    try:
        import wandb

        if wandb.run is None:
            return
        artifact = wandb.Artifact(
            f"{split}-trajectory-step-{step}",
            type="validation-trajectory" if validate else "train-trajectory",
        )
        table = wandb.Table(columns=["step", "score", "path", "preview"])
        for idx, (path, score) in enumerate(samples):
            segment_paths = [path]
            if os.path.isdir(path):
                segment_paths = [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".txt")]

            preview_parts = []
            for segment_index, segment_path in enumerate(segment_paths):
                artifact.add_file(
                    segment_path,
                    name=f"{idx:03d}_{segment_index:03d}_{os.path.basename(segment_path)}",
                )
                with open(segment_path, encoding="utf-8") as file:
                    preview_parts.append(f"===== {os.path.basename(segment_path)} =====\n{file.read(20000)}")
            preview = "\n\n".join(preview_parts)[:20000]
            table.add_data(step, score, path, preview)

        wandb.log_artifact(artifact)
        wandb.log({table_key: table}, step=step)
    except Exception as exc:
        print(f"Warning: failed to log {table_key} to wandb: {exc}")


def log_train_trajectories_to_wandb(batch, trainer, step: int) -> None:
    """Read saved paths and task scores without changing the training batch."""
    if int(trainer.get("log_train_trajectory_txt", 0) or 0) <= 0:
        return
    if "gym_trajectory_txt_path" not in batch.non_tensor_batch:
        print("Warning: log_train_trajectory_txt > 0, but no gym trajectory txt paths were returned.")
        return

    paths = batch.non_tensor_batch["gym_trajectory_txt_path"]
    paths = paths.tolist() if hasattr(paths, "tolist") else list(paths)
    if "token_level_scores" in batch.batch:
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
    else:
        scores = [None] * len(paths)
    log_trajectories_to_wandb(paths, scores, trainer, step, validate=False)
