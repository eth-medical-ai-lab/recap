"""Build the reusable E2B templates required by SWE rollout datasets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.envs.swe_agent.tasks import E2B_CPU_COUNT, E2B_MEMORY_MB, e2b_template_name

LEDGER_VERSION = 1


def load_template_specs(
    source_files: list[Path],
    requested_task_ids: set[str] | None = None,
    *,
    cpu_count: int = E2B_CPU_COUNT,
    memory_mb: int = E2B_MEMORY_MB,
) -> list[dict[str, Any]]:
    """Collect one template spec per distinct image from JSONL task files."""
    by_image: dict[str, dict[str, Any]] = {}
    found_task_ids: set[str] = set()
    for source_file in source_files:
        with source_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {source_file}:{line_number}") from error
                task_id, image_name, supplied_template = _task_fields(
                    row,
                    source_file,
                    line_number,
                )
                if requested_task_ids is not None and task_id not in requested_task_ids:
                    continue

                template_name = e2b_template_name(image_name, cpu_count=cpu_count, memory_mb=memory_mb)
                if supplied_template is not None and supplied_template != template_name:
                    raise ValueError(f"Unexpected template_name for {task_id!r} at {source_file}:{line_number}")
                found_task_ids.add(task_id)
                spec = by_image.setdefault(
                    image_name,
                    {
                        "image_name": image_name,
                        "template_name": template_name,
                        "task_ids": [],
                    },
                )
                spec["task_ids"].append(task_id)

    if requested_task_ids is not None:
        missing = requested_task_ids - found_task_ids
        if missing:
            raise ValueError(f"Requested task IDs not found: {sorted(missing)}")
    if not by_image:
        raise ValueError("No SWE task images found")

    specs = list(by_image.values())
    for spec in specs:
        spec["task_ids"] = sorted(set(spec["task_ids"]))
    return sorted(specs, key=lambda spec: spec["template_name"])


def build_templates(
    specs: list[dict[str, Any]],
    ledger_path: Path,
    *,
    template_api: Any,
    build_logger: Any,
    cpu_count: int,
    memory_mb: int,
    force: bool,
    sdk_version: str,
) -> dict[str, Any]:
    """Build missing templates and save progress after every image."""
    ledger = _load_ledger(ledger_path)
    ledger["e2b_sdk_version"] = sdk_version
    failures = []

    for spec in specs:
        image_name = spec["image_name"]
        template_name = spec["template_name"]
        # An image-only or differently sized alias must never be reused and
        # recorded as having the requested resources, even with --force.
        expected_name = e2b_template_name(image_name, cpu_count=cpu_count, memory_mb=memory_mb)
        if template_name != expected_name:
            raise ValueError(f"Template name does not match requested resources: {template_name}")
        previous = ledger["templates"].get(image_name, {})
        attempt = int(previous.get("attempt", 0)) + 1
        started_at = time.monotonic()
        record = {
            **spec,
            "attempt": attempt,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
            "updated_at": _utc_now(),
        }
        try:
            exists = not force and template_api.exists(template_name)
            if exists:
                record.update({"status": "ready", "origin": "existing"})
                # Preserve the original build identifiers when a resumed run
                # confirms that the template is still available.
                for field in ("template_id", "build_id"):
                    if field in previous:
                        record[field] = previous[field]
            else:
                builder = template_api().from_image(image_name).set_workdir("/testbed")
                if image_name == "jyangballin/swesmith.x86_64.pandas-dev_1776_pandas.95280573":
                    # Pandas uses a Meson editable install. Its first import
                    # in E2B rebuilds native extensions; compile the clean
                    # baseline once here instead of during every episode.
                    builder = builder.run_cmd(
                        "set -eu\n"
                        "export PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH\n"
                        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                        'git checkout --detach "$(git rev-list --max-parents=0 HEAD)"\n'
                        "python -c 'import pandas; print(pandas.__version__)'",
                        user="root",
                    )
                elif image_name == "jyangballin/swesmith.x86_64.pydicom_1776_pydicom.7d361b3d":
                    # This public regression fixture is downloaded lazily by
                    # upstream tests. Bundle it while builds have networking,
                    # in the package data directory readable by both users.
                    builder = builder.run_cmd(
                        "set -eu\n"
                        "export PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH\n"
                        'git checkout --detach "$(git rev-list --max-parents=0 HEAD)"\n'
                        "python -c 'from pathlib import Path; import shutil; "
                        "from pydicom.data.download import data_path_with_download; "
                        "from pydicom.data.data_manager import DATA_ROOT; "
                        'name="MR-SIEMENS-DICOM-WithOverlays.dcm"; '
                        "source=data_path_with_download(name); "
                        'shutil.copyfile(source, Path(DATA_ROOT)/"test_files"/name)\'',
                        user="root",
                    )
                build_info = template_api.build(
                    builder,
                    template_name,
                    cpu_count=cpu_count,
                    memory_mb=memory_mb,
                    skip_cache=force,
                    on_build_logs=build_logger,
                )
                record.update(
                    {
                        "status": "ready",
                        "origin": "built",
                        "template_id": build_info.template_id,
                        "build_id": build_info.build_id,
                    }
                )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            failures.append(template_name)
        record["duration_seconds"] = time.monotonic() - started_at
        ledger["templates"][image_name] = record
        _write_ledger(ledger_path, ledger)
        print(
            f"{template_name}: {record['status']} ({record['duration_seconds']:.1f}s)",
            flush=True,
        )

    if failures:
        raise RuntimeError(f"Failed E2B templates: {failures}")
    return ledger


def _task_fields(row: Any, source_file: Path, line_number: int) -> tuple[str, str, str | None]:
    if not isinstance(row, dict):
        raise ValueError(f"Task row must be an object at {source_file}:{line_number}")
    extra_info = row.get("extra_info")
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    task_id = row.get("instance_id") or extra_info.get("task_id")
    image_name = row.get("image_name") or row.get("image") or extra_info.get("image_name")
    template_name = row.get("template_name") or extra_info.get("template_name")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Missing task ID at {source_file}:{line_number}")
    if not isinstance(image_name, str) or not image_name:
        raise ValueError(f"Missing image name at {source_file}:{line_number}")
    if template_name is not None and (not isinstance(template_name, str) or not template_name):
        raise ValueError(f"Invalid template name at {source_file}:{line_number}")
    return task_id, image_name, template_name


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": LEDGER_VERSION, "templates": {}}
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("version") != LEDGER_VERSION or not isinstance(ledger.get("templates"), dict):
        raise ValueError(f"Invalid E2B template ledger: {path}")
    return ledger


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_files", nargs="+", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--cpu-count", type=int, default=E2B_CPU_COUNT)
    parser.add_argument("--memory-mb", type=int, default=E2B_MEMORY_MB)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cpu_count <= 0 or args.memory_mb <= 0:
        raise ValueError("cpu-count and memory-mb must be positive")

    from e2b import Template, default_build_logger

    specs = load_template_specs(
        args.source_files,
        set(args.task_ids) if args.task_ids else None,
        cpu_count=args.cpu_count,
        memory_mb=args.memory_mb,
    )
    ledger = build_templates(
        specs,
        args.ledger,
        template_api=Template,
        build_logger=default_build_logger(),
        cpu_count=args.cpu_count,
        memory_mb=args.memory_mb,
        force=args.force,
        sdk_version=importlib.metadata.version("e2b"),
    )
    ledger["sources"] = [{"path": str(path.resolve())} for path in args.source_files]
    _write_ledger(args.ledger, ledger)


if __name__ == "__main__":
    main()
