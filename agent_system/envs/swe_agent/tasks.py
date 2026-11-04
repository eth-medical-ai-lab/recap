"""SWE-Smith and SWE-bench task loading, image names, and solution records."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
HARNESS_REVISION = "7a21e05772954cc81471ae19d56f436cecf43c54"
SPLIT = "test"
_CACHE_SIZE = 16
_INSTANCE_ID_RE = re.compile(rb'"instance_id"\s*:\s*("(?:\\.|[^"\\])*")')
_ROLLOUT_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
_EVALUATOR_FIELDS = (
    "image",
    "version",
    "patch",
    "test_patch",
    "eval_script",
    "log_parser",
    "eval_type",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


# Image/resource identity used by task rows and template preparation.
TEMPLATE_PREFIX = "swe-agent-"
E2B_CPU_COUNT = 1
E2B_MEMORY_MB = 2048


def e2b_template_name(image_name: str, *, cpu_count: int = E2B_CPU_COUNT, memory_mb: int = E2B_MEMORY_MB) -> str:
    """Bind an image and its sandbox resources to one stable template name."""
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError("image_name must be a nonempty string")
    if type(cpu_count) is not int or cpu_count <= 0 or type(memory_mb) is not int or memory_mb <= 0:
        raise ValueError("cpu_count and memory_mb must be positive integers")
    digest = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:24]
    return f"{TEMPLATE_PREFIX}{cpu_count}c-{memory_mb}m-{digest}"


# SWE-Smith: index JSONL by byte offset and decode only requested tasks.
@lru_cache(maxsize=8)
def swe_smith_task_offsets(data_file: str) -> dict[str, tuple[int, int]]:
    """Index IDs without retaining complete task records in RAM."""
    offsets = {}
    with Path(data_file).open("rb") as file:
        line_number = 0
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                match = _INSTANCE_ID_RE.search(line)
                instance_id = json.loads(match.group(1)) if match is not None else None
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid instance_id at {data_file}:{line_number}.") from error
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"Missing instance_id at {data_file}:{line_number}.")
            if instance_id in offsets:
                raise ValueError(f"Duplicate SWE-smith instance_id {instance_id!r} in {data_file}.")
            offsets[instance_id] = (offset, line_number)

    if not offsets:
        raise ValueError(f"SWE-smith task file is empty: {data_file}")
    return offsets


@lru_cache(maxsize=16)
def load_swe_smith_task_at(data_file: str, offset: int, line_number: int) -> dict[str, Any]:
    with Path(data_file).open("rb") as file:
        file.seek(offset)
        line = file.readline()
    try:
        task = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON at {data_file}:{line_number}.") from error
    if not isinstance(task, dict):
        raise ValueError(f"SWE-smith task must be an object at {data_file}:{line_number}.")

    instance_id = _nonempty_string(task.get("instance_id"))
    problem = _nonempty_string(task.get("problem_statement"))
    image = _nonempty_string(task.get("image_name"))
    repo = _nonempty_string(task.get("repo"))
    fail_to_pass = _swe_smith_test_list(task.get("FAIL_TO_PASS"), allow_empty=False)
    pass_to_pass = _swe_smith_test_list(task.get("PASS_TO_PASS"), allow_empty=True)
    if None in (instance_id, problem, image, repo) or fail_to_pass is None or pass_to_pass is None:
        raise ValueError(f"Invalid SWE-smith task at {data_file}:{line_number}.")
    expected_template = e2b_template_name(image)
    supplied_template = task.get("template_name")
    if supplied_template is not None and supplied_template != expected_template:
        raise ValueError(f"Invalid E2B template at {data_file}:{line_number}.")
    task["template_name"] = expected_template
    return task


def load_swe_smith_task(data_file: str, instance_id: str) -> dict[str, Any]:
    location = swe_smith_task_offsets(data_file).get(instance_id)
    if location is None:
        raise KeyError(f"Unknown SWE-smith instance_id {instance_id!r} in {data_file}.")
    task = load_swe_smith_task_at(data_file, *location)
    if task["instance_id"] != instance_id:
        raise ValueError(f"SWE-smith task at {data_file}:{location[1]} has mismatched ID.")
    return task


def resolve_swe_smith_data_file(data_path: str, split: str) -> str:
    path = Path(data_path).expanduser()
    if path.is_dir():
        path = path / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"SWE-smith task file not found: {path}")
    return str(path.resolve())


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _swe_smith_test_list(value: Any, *, allow_empty: bool) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return None
    if not allow_empty and not value:
        return None
    return value


# SWE-bench Verified: separate model-visible rows from private test metadata.
class SweBenchVerifiedTaskStore:
    """Expose safe rollout rows while keeping the complete evaluator row private.

    JSONL files are indexed by byte offset and only selected rows are decoded.
    A JSON array is also supported because Verified has only 500 evaluation rows.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path).expanduser().resolve()
        if not self.data_path.is_file():
            raise FileNotFoundError(f"SWE-bench Verified task file not found: {self.data_path}")

        self._json_rows: dict[str, dict[str, Any]] | None = None
        self._offsets: dict[str, tuple[int, int]] | None = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        if self.data_path.suffix.lower() == ".jsonl":
            self._offsets = self._index_jsonl()
        elif self.data_path.suffix.lower() == ".json":
            self._json_rows = self._load_json_array()
        else:
            raise ValueError("SWE-bench Verified data_path must end in .json or .jsonl")

    @property
    def task_ids(self) -> tuple[str, ...]:
        rows = self._offsets if self._offsets is not None else self._json_rows
        assert rows is not None
        return tuple(rows)

    def list_task_specs(self) -> list[dict[str, Any]]:
        """Return scheduler metadata with no problem or evaluator-only fields."""
        specs = []
        for instance_id in self.task_ids:
            row = self._load(instance_id)
            specs.append(
                {
                    "env_name": "swe_agent",
                    "dataset": DATASET_NAME,
                    "split": SPLIT,
                    "evaluation_only": True,
                    "task_id": instance_id,
                    "uid": f"swe_agent:{SPLIT}:{instance_id}",
                    "instance_id": instance_id,
                    "repo": row["repo"],
                    "base_commit": row["base_commit"],
                    "image_name": row["image"],
                    "template_name": e2b_template_name(row["image"]),
                }
            )
        return specs

    def load_rollout_task(self, instance_id: str) -> dict[str, Any]:
        """Return only fields that may be passed to the model-facing environment."""
        row = self._load(instance_id)
        return {
            "dataset": DATASET_NAME,
            "split": SPLIT,
            "evaluation_only": True,
            **{field: row[field] for field in _ROLLOUT_FIELDS},
            "image_name": row["image"],
            "template_name": e2b_template_name(row["image"]),
        }

    def load_private_task(self, instance_id: str) -> dict[str, Any]:
        """Return the complete row for the hidden evaluator process only."""
        row = copy.deepcopy(self._load(instance_id))
        row.update(
            {
                "dataset": DATASET_NAME,
                "dataset_revision": DATASET_REVISION,
                "harness_revision": HARNESS_REVISION,
                "split": SPLIT,
                "evaluation_only": True,
                "image_name": row["image"],
                "template_name": e2b_template_name(row["image"]),
                "dataset_type": "swebench_verified",
                "checkout_ref": row["base_commit"],
            }
        )
        return row

    def _load(self, instance_id: str) -> dict[str, Any]:
        if instance_id in self._cache:
            self._cache.move_to_end(instance_id)
            return self._cache[instance_id]

        if self._json_rows is not None:
            try:
                row = self._json_rows[instance_id]
            except KeyError as error:
                raise KeyError(f"Unknown SWE-bench Verified instance_id {instance_id!r}") from error
        else:
            assert self._offsets is not None
            try:
                offset, line_number = self._offsets[instance_id]
            except KeyError as error:
                raise KeyError(f"Unknown SWE-bench Verified instance_id {instance_id!r}") from error
            with self.data_path.open("rb") as file:
                file.seek(offset)
                line = file.readline()
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid JSON at {self.data_path}:{line_number}") from error

        row = _validate_row(row, str(self.data_path))
        if row["instance_id"] != instance_id:
            raise ValueError(f"SWE-bench Verified row at {self.data_path} has mismatched instance_id")
        self._cache[instance_id] = row
        self._cache.move_to_end(instance_id)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return row

    def _index_jsonl(self) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        with self.data_path.open("rb") as file:
            line_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    match = _INSTANCE_ID_RE.search(line)
                    instance_id = json.loads(match.group(1)) if match is not None else None
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"Invalid instance_id at {self.data_path}:{line_number}") from error
                _add_unique(offsets, instance_id, (offset, line_number), str(self.data_path))
        if not offsets:
            raise ValueError(f"SWE-bench Verified task file is empty: {self.data_path}")
        return offsets

    def _load_json_array(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"Invalid JSON in {self.data_path}") from error
        if not isinstance(payload, list) or not payload:
            raise ValueError("SWE-bench Verified .json input must be a non-empty array")

        rows: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(payload, start=1):
            row = _validate_row(row, f"{self.data_path}[{index}]")
            _add_unique(rows, row["instance_id"], row, str(self.data_path))
        return rows


def swebench_verified_image(instance_id: str) -> str:
    """Return the official remote x86_64 instance image key."""
    instance_id = _required_string(instance_id, "instance_id", "image name")
    key = f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
    return key.replace("__", "_1776_")


def make_swebench_solution(
    instance_id: str,
    model_patch: str,
    model_name_or_path: str,
) -> dict[str, str]:
    """Build one record accepted by the official SWE-bench evaluator."""
    return {
        "instance_id": _required_string(instance_id, "instance_id", "solution"),
        "model_patch": _string(model_patch, "model_patch", "solution"),
        "model_name_or_path": _required_string(model_name_or_path, "model_name_or_path", "solution"),
    }


def _validate_row(row: Any, location: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"SWE-bench Verified task must be an object at {location}")
    for field in _ROLLOUT_FIELDS:
        _required_string(row.get(field), field, location)
    for field in _EVALUATOR_FIELDS:
        if field in {"FAIL_TO_PASS", "PASS_TO_PASS"}:
            _test_list(row.get(field), field, location)
        else:
            _required_string(row.get(field), field, location)
    expected_image = swebench_verified_image(row["instance_id"])
    if row["image"] != expected_image:
        raise ValueError(f"image must be the official x86_64 instance image at {location}: expected {expected_image!r}")
    if "split" in row and row["split"] != SPLIT:
        raise ValueError(f"SWE-bench Verified records are evaluation-only at {location}")
    if "evaluation_only" in row and row["evaluation_only"] is not True:
        raise ValueError(f"SWE-bench Verified records are evaluation-only at {location}")
    return row


def _test_list(value: Any, field: str, location: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} must contain a JSON list at {location}") from error
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a list of nonempty strings at {location}")
    return value


def _add_unique(mapping: dict, instance_id: Any, value: Any, location: str) -> None:
    instance_id = _required_string(instance_id, "instance_id", location)
    if instance_id in mapping:
        raise ValueError(f"Duplicate SWE-bench Verified instance_id {instance_id!r} in {location}")
    mapping[instance_id] = value


def _required_string(value: Any, field: str, location: str) -> str:
    value = _string(value, field, location)
    if not value.strip():
        raise ValueError(f"{field} must be non-empty at {location}")
    return value


def _string(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string at {location}")
    return value
