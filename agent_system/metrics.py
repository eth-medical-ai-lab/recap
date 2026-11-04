"""Episode and GRPO diagnostics, counting each real trajectory once across segments."""

from collections import defaultdict

import numpy as np

from verl import DataProto

_EPISODE_STATS = (
    "reward",
    "length",
    "tool_call_count",
    "response_length",
    "token_budget",
    "token_count",
    "action_token_count",
    "observation_token_count",
    "summary_token_count",
    "instruction_token_count",
)
_EPISODE_RATES = {
    "episode_success": "success_rate",
    "episode_done": "done_rate",
    "episode_terminated": "terminated_rate",
    "episode_truncated": "truncated_rate",
    "episode_answer_submitted": "submission_rate",
    "turn_token_limit_hit": "turn_token_limit_rate",
    "context_length_hit": "context_length_limit_rate",
    "episode_token_limit_hit": "token_limit_rate",
}
EPISODE_FIELD_KEYS = (
    *(f"episode_{name}" for name in _EPISODE_STATS),
    *_EPISODE_RATES,
    "finish_reason",
    "episode_step_response_length_mean",
    "episode_step_response_length_min",
    "episode_step_response_length_max",
)
FORMAT_FIELD_KEYS = ("format_ok", "format_ok_rate", "invalid_format_count")
FINISH_REASONS = (
    "done",
    "max_assistant_turns",
    "response_length",
    "episode_token_budget",
    "context_length",
    "invalid_action",
    "max_compactions",
    "compaction_capacity",
    "rollout_timeout",
    "environment_error",
)


def _episode_rows(fields: dict) -> list[int]:
    """Select one non-padding row per trajectory; ordinary rollout rows stand alone."""
    row_count = len(next(iter(fields.values()), ()))
    trajectory_ids = fields.get("compaction_trajectory_uid")
    padding = fields.get("compaction_padding", np.zeros(row_count, dtype=bool))
    rows, seen = [], set()
    for row in range(row_count):
        if padding[row]:
            continue
        trajectory_id = trajectory_ids[row] if trajectory_ids is not None else None
        if trajectory_id is not None:
            if trajectory_id in seen:
                continue
            seen.add(trajectory_id)
        rows.append(row)
    return rows


def _numeric_values(fields: dict, key: str, rows: list[int]) -> np.ndarray:
    """Optional scalar fields use None for unavailable values, such as SWE submission."""
    if key not in fields:
        return np.array([], dtype=np.float32)
    return np.asarray([fields[key][row] for row in rows if fields[key][row] is not None], dtype=np.float32)


def _grpo_group_metrics(fields: dict, rows: list[int]) -> dict[str, float]:
    """Group already-deduplicated binary outcomes by prompt."""
    if not {"uid", "compaction_trajectory_uid", "episode_success"}.issubset(fields):
        return {}
    groups = defaultdict(list)
    for row in rows:
        group_id = fields["uid"][row]
        success = fields["episode_success"][row]
        if group_id is not None and fields["compaction_trajectory_uid"][row] is not None and success is not None:
            groups[group_id].append(float(success))
    if not groups:
        return {}

    correct = np.array([sum(value >= 0.5 for value in successes) for successes in groups.values()])
    sizes = np.array([len(successes) for successes in groups.values()])
    metrics = {
        "grpo/group/count": len(groups),
        "grpo/group/size_mean": float(sizes.mean()),
        "grpo/group/size_min": int(sizes.min()),
        "grpo/group/size_max": int(sizes.max()),
        "grpo/group/all_wrong_rate": float((correct == 0).mean()),
        "grpo/group/all_correct_rate": float((correct == sizes).mean()),
        "grpo/group/mixed_rate": float(((correct > 0) & (correct < sizes)).mean()),
    }
    for count in range(int(sizes.max()) + 1):
        metrics[f"grpo/group/correct_count_{count}_rate"] = float((correct == count).mean())
    return metrics


def compute_episode_metrics(batch: DataProto, prefix: str = "episode") -> dict[str, float]:
    """Read repeated episode scalars; segment splitting and padding must not change their weights."""
    fields = batch.non_tensor_batch
    rows = _episode_rows(fields)
    metrics = {}

    # Episode totals and outcomes all use the same real trajectory rows.
    for name in _EPISODE_STATS:
        values = _numeric_values(fields, f"episode_{name}", rows)
        if values.size:
            metrics[f"{prefix}/{name}/mean"] = float(values.mean())
            metrics[f"{prefix}/{name}/max"] = float(values.max())
            metrics[f"{prefix}/{name}/min"] = float(values.min())
    for field, name in _EPISODE_RATES.items():
        values = _numeric_values(fields, field, rows)
        if values.size:
            metrics[f"{prefix}/{name}"] = float(values.mean())
    for name, reduce in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        values = _numeric_values(fields, f"episode_step_response_length_{name}", rows)
        if values.size:
            metrics[f"{prefix}/step_response_length/{name}"] = float(reduce(values))
    reasons = fields.get("finish_reason")
    if reasons is not None:
        reasons = [reasons[row] for row in rows if reasons[row] is not None]
        if reasons:
            for reason in FINISH_REASONS:
                metrics[f"{prefix}/finish_reason/{reason}_rate"] = float(
                    np.mean([value == reason for value in reasons])
                )

    # Keep existing metric names. "mean_compact_time" is a count, not a duration.
    counts = _numeric_values(fields, "compaction_count", rows)
    if counts.size:
        metrics["mean_compact_time"] = float(counts.mean())
    if "compaction_count" in fields:
        summary_tokens = 0
        summary_count = 0
        grouped_successes = defaultdict(list)
        for row in rows:
            count = fields["compaction_count"][row]
            if count is None:
                continue
            if "episode_summary_token_count" in fields and fields["episode_summary_token_count"][row] is not None:
                summary_tokens += fields["episode_summary_token_count"][row]
                summary_count += count
            if "episode_success" in fields and fields["episode_success"][row] is not None:
                grouped_successes[int(count)].append(float(fields["episode_success"][row]))
        # Average over summaries, preserving the old flattened-list weighting.
        if summary_count:
            metrics["mean_compact_response_tokens"] = float(summary_tokens / summary_count)
        for count, successes in sorted(grouped_successes.items()):
            metrics[f"{prefix}/success_rate/compactions_{count}"] = float(np.mean(successes))
            metrics[f"{prefix}/count/compactions_{count}"] = len(successes)

    metrics.update(_grpo_group_metrics(fields, rows))
    return metrics


def compute_format_metrics(batch: DataProto, prefix: str = "format") -> dict[str, float]:
    """Average per-episode format statistics, excluding duplicate segments and padding."""
    fields = batch.non_tensor_batch
    rows = _episode_rows(fields)
    metrics = {}
    for field, name in (("format_ok", "ok_rate"), ("format_ok_rate", "step_ok_rate")):
        values = _numeric_values(fields, field, rows)
        if values.size:
            metrics[f"{prefix}/{name}"] = float(values.mean())
    values = _numeric_values(fields, "invalid_format_count", rows)
    if values.size:
        metrics[f"{prefix}/invalid_count_mean"] = float(values.mean())
        metrics[f"{prefix}/invalid_count_max"] = float(values.max())
    return metrics
