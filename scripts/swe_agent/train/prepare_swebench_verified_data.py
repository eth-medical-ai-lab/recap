"""Download all 500 SWE-bench Verified tasks and prepare test rollout rows."""

import argparse
import json
from pathlib import Path

from agent_system.envs.swe_agent.tasks import DATASET_NAME, DATASET_REVISION, e2b_template_name


def prepare(tasks: list[dict], output_dir: Path) -> None:
    """Keep complete tasks private to the environment and expose only rollout metadata."""
    if len(tasks) != 500 or len({task["instance_id"] for task in tasks}) != 500:
        raise ValueError("Expected 500 unique SWE-bench Verified tasks")
    output_dir = output_dir.resolve()
    task_file = output_dir / "tasks/test.jsonl"
    rows = []
    for index, task in enumerate(tasks):
        task_id = task["instance_id"]
        rows.append(
            {
                "data_source": "swebench_verified",
                "agent_name": "gym_agent",
                "prompt": [{"role": "user", "content": "SweAgentEnv loads the task prompt privately."}],
                "ability": "swe_agent",
                "reward_model": {"style": "rule", "ground_truth": task_id},
                "extra_info": {
                    "env_name": "swe_agent",
                    "split": "test",
                    "task_id": task_id,
                    "image_name": task["image"],
                    "template_name": e2b_template_name(task["image"]),
                    "repo": task["repo"],
                    "index": index,
                    "env_kwargs": {"dataset": "swebench_verified", "data_path": task_file.as_posix()},
                },
                "uid": f"swe_agent:test:{task_id}",
            }
        )

    # The pinned official dataset already includes images and evaluator fields.
    for name, records in {"tasks/test.jsonl": tasks, "rollouts/test.jsonl": rows}.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/swe_agent"))
    args = parser.parse_args()
    from datasets import load_dataset

    tasks = list(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test"))
    prepare(tasks, args.output_dir)
    print(f"Prepared 500 test tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
