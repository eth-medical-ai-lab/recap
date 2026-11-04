"""Prepare the final 300 SWE-Smith tasks and eight ordered training passes."""

import argparse
import json
import random
from pathlib import Path

from agent_system.envs.swe_agent.tasks import e2b_template_name

DATASET_NAME = "SWE-bench/SWE-smith"
DATASET_REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"
SELECTION_FILE = Path(__file__).with_name("train_300_selection.json")
TASK_FIELDS = ("instance_id", "problem_statement", "image_name", "repo", "FAIL_TO_PASS", "PASS_TO_PASS")
EPOCH_SEEDS = list(range(44, 52))


def select_tasks(source, task_ids: list[str]) -> list[dict]:
    """Read only the selected tasks into memory and restore their fixed order."""
    selected = set(task_ids)
    if len(task_ids) != 300 or len(selected) != 300:
        raise ValueError("Expected 300 unique training task IDs")
    tasks = {}
    for task in source:
        if task["instance_id"] in selected:
            tasks[task["instance_id"]] = {field: task[field] for field in TASK_FIELDS}
            if len(tasks) == len(selected):
                break
    missing = selected - tasks.keys()
    if missing:
        raise ValueError(f"Selected tasks missing from SWE-Smith: {sorted(missing)}")
    return [tasks[task_id] for task_id in task_ids]


def prepare(tasks: list[dict], output_dir: Path) -> None:
    """Preserve epoch order and keep private test lists out of rollout rows."""
    output_dir = output_dir.resolve()
    task_file = output_dir / "tasks/train.jsonl"
    rows = []
    for epoch, seed in enumerate(EPOCH_SEEDS, start=1):
        epoch_tasks = tasks.copy()
        random.Random(seed).shuffle(epoch_tasks)
        for task in epoch_tasks:
            task_id = task["instance_id"]
            index = len(rows)
            rows.append(
                {
                    "data_source": "swe_smith",
                    "agent_name": "gym_agent",
                    "ability": "swe_agent",
                    "prompt": [{"role": "user", "content": "SweAgentEnv loads the task prompt privately."}],
                    "reward_model": {"style": "rule", "ground_truth": task_id},
                    # Each visit remains a separate GRPO group.
                    "uid": f"swe_smith:train:epoch{epoch}:visit{index}:{task_id}",
                    "extra_info": {
                        "env_name": "swe_agent",
                        "split": "train",
                        "task_id": task_id,
                        "index": index,
                        "epoch": epoch,
                        "image_name": task["image_name"],
                        "template_name": e2b_template_name(task["image_name"]),
                        "env_kwargs": {"dataset": "swe_smith", "data_path": task_file.as_posix()},
                    },
                }
            )

    for name, records in {"tasks/train.jsonl": tasks, "rollouts/train.jsonl": rows}.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/swe_agent"))
    args = parser.parse_args()
    from datasets import load_dataset

    task_ids = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))["task_ids"]
    source = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
        columns=list(TASK_FIELDS),
        filters=[("instance_id", "in", task_ids)],
    )
    tasks = select_tasks(source, task_ids)
    prepare(tasks, args.output_dir)
    print(f"Prepared 300 training tasks and 2,400 rollout rows in {args.output_dir}")


if __name__ == "__main__":
    main()
