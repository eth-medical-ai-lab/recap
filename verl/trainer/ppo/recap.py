"""RECAP segment batches, task/summary credit, and continuation self-distillation."""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def recap_coefficients(config) -> tuple[float, float]:
    """Resolve shared/individual weights; zero overrides the shared coefficient."""
    coef = float(config.get("coef", 0.1))
    summary = config.get("summary_pg_coef")
    continuation = config.get("continuation_sd_coef")
    summary = coef if summary is None else float(summary)
    continuation = coef if continuation is None else float(continuation)
    if min(coef, summary, continuation) < 0:
        raise ValueError("RECAP coefficients must be non-negative.")
    return summary, continuation


def recap_needs_scoring(config, step: int | None = None) -> bool:
    """Rollouts prepare every possible measurement; the trainer selects updates."""
    if not config.get("enabled", False):
        return False
    summary, continuation = recap_coefficients(config)
    interval = int(config.get("distortion_measure_interval", 0))
    return summary > 0 or continuation > 0 or (interval > 0 and (step is None or step % interval == 0))


def unpad_generation_output(data: DataProto, original_len: int, pad_size: int) -> DataProto:
    """Remove padded parent samples after rollout.

    Variable-output rollouts flatten several rows per parent. The parent index
    lets us drop every row that came from a padded parent sample.
    """
    parent = data.non_tensor_batch.get("rollout_parent_index")
    if pad_size == 0 or parent is None:
        return unpad_dataproto(data, pad_size=pad_size)

    parent = np.asarray(parent, dtype=np.int64)
    return data.select_idxs(parent < original_len)


def align_batch_to_generation_output(batch: DataProto, gen_output: DataProto) -> tuple[DataProto, DataProto]:
    """Align original parent rows with generated rows before `DataProto.union`.

    Normal rollouts have one generated row per parent row. Variable-output
    rollouts use `rollout_parent_index` to select the parent for each result.
    """
    parent = gen_output.non_tensor_batch.get("rollout_parent_index")
    if parent is None:
        if len(batch) != len(gen_output):
            raise ValueError(f"Generation output length mismatch: {len(batch)=}, {len(gen_output)=}")
        return batch, gen_output

    parent = np.asarray(parent, dtype=np.int64)
    batch = batch.select_idxs(parent)

    overlap_keys = sorted(set(batch.non_tensor_batch) & set(gen_output.non_tensor_batch))
    if overlap_keys:
        for key in overlap_keys:
            batch.non_tensor_batch[key] = gen_output.non_tensor_batch[key]
        gen_output.pop(non_tensor_batch_keys=overlap_keys)

    return batch, gen_output


def pad_compaction_batch_for_data_parallel(batch: DataProto, divisor: int) -> DataProto:
    """Pad variable-segment compaction batches without adding trainable samples."""
    if "compaction_trajectory_uid" not in batch.non_tensor_batch:
        return batch

    batch.non_tensor_batch["compaction_padding"] = np.zeros(len(batch), dtype=bool)

    # Metadata describes the logical batch and can contain NumPy arrays. Keep it
    # out of row padding because DataProto.concat expects scalar metadata values.
    meta_info = batch.meta_info
    batch.meta_info = {}
    batch, pad_size = pad_dataproto_to_divisor(batch, divisor)
    batch.meta_info = meta_info
    if pad_size == 0:
        return batch

    batch.non_tensor_batch["compaction_padding"][-pad_size:] = True
    batch.batch["response_mask"][-pad_size:] = 0
    if "rm_scores" in batch.batch:
        batch.batch["rm_scores"][-pad_size:] = 0

    return batch


def remove_compaction_padding(
    batch: DataProto, reward_extra_infos: dict[str, list]
) -> tuple[DataProto, dict[str, list]]:
    """Drop data-parallel padding before trajectory logging and metrics."""
    padding = batch.non_tensor_batch.get("compaction_padding")
    if padding is None:
        return batch, reward_extra_infos

    keep = ~np.asarray(padding, dtype=bool)
    reward_extra_infos = {
        key: [value for value, keep_row in zip(values, keep, strict=True) if keep_row]
        for key, values in reward_extra_infos.items()
    }
    batch = batch.select_idxs(keep)
    batch.pop(non_tensor_batch_keys=["compaction_padding"])
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    return batch, reward_extra_infos


def compute_grpo_trajectory_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_index: np.ndarray,
    padding_mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    empty_failure_mask: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute one GRPO advantage and one token-mean loss weight per logical trajectory.

    A compacted rollout may occupy several model rows with different prompts.
    The rows share one outcome reward, and their token weights sum to one so a
    long or frequently compacted trajectory does not receive extra loss weight.
    """
    scores = token_level_rewards.sum(dim=-1)
    padding_mask = np.zeros(len(scores), dtype=bool) if padding_mask is None else np.asarray(padding_mask, dtype=bool)
    if len(index) != len(scores) or len(trajectory_index) != len(scores) or len(padding_mask) != len(scores):
        raise ValueError("GRPO trajectory metadata must have one entry per model row.")

    trajectory_rows = defaultdict(list)
    for row in range(len(scores)):
        if not padding_mask[row]:
            trajectory_rows[trajectory_index[row]].append(row)
    if not trajectory_rows:
        raise ValueError("GRPO trajectory batch contains no non-padding rows.")

    trajectory_scores = {}
    trajectory_groups = {}
    group_scores = defaultdict(list)
    with torch.no_grad():
        for trajectory_id, rows in trajectory_rows.items():
            groups = {index[row] for row in rows}
            if len(groups) != 1:
                raise ValueError(f"Logical trajectory {trajectory_id!r} spans multiple GRPO prompt groups.")

            row_scores = torch.stack([scores[row] for row in rows])
            if not torch.allclose(row_scores, row_scores[0].expand_as(row_scores)):
                raise ValueError(f"Logical trajectory {trajectory_id!r} has inconsistent segment rewards.")

            group_id = index[rows[0]]
            trajectory_scores[trajectory_id] = row_scores[0]
            trajectory_groups[trajectory_id] = group_id
            group_scores[group_id].append(row_scores[0])

        group_stats = {}
        for group_id, values in group_scores.items():
            values = torch.stack(values)
            if len(values) == 1:
                group_stats[group_id] = (values.new_tensor(0.0), values.new_tensor(1.0))
            else:
                group_stats[group_id] = (values.mean(), values.std())

        advantages = torch.zeros_like(token_level_rewards)
        loss_weights = torch.zeros_like(token_level_rewards)
        for trajectory_id, rows in trajectory_rows.items():
            group_id = trajectory_groups[trajectory_id]
            mean, std = group_stats[group_id]
            advantage = trajectory_scores[trajectory_id] - mean
            if norm_adv_by_std_in_grpo:
                advantage = advantage / (std + epsilon)

            token_count = response_mask[rows].sum()
            if token_count.item() <= 0:
                if empty_failure_mask is not None and all(empty_failure_mask[row] for row in rows):
                    # Keep this zero outcome in the eight-way group statistics,
                    # but there are no generated tokens to train on.
                    continue
                raise ValueError(f"Logical trajectory {trajectory_id!r} has no trainable tokens.")
            for row in rows:
                advantages[row] = advantage * response_mask[row]
                loss_weights[row] = response_mask[row] / token_count

    return advantages, advantages, loss_weights, len(trajectory_rows)


def compute_grpo_segment_advantage(data: DataProto, norm_adv_by_std_in_grpo: bool):
    """Normalize task rewards over trajectories and share their credit across segments."""
    advantages, returns, loss_weights, trajectory_count = compute_grpo_trajectory_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        trajectory_index=data.non_tensor_batch["compaction_trajectory_uid"],
        padding_mask=data.non_tensor_batch.get("compaction_padding"),
        empty_failure_mask=np.logical_or(
            data.non_tensor_batch.get("swe_infrastructure_failure", np.zeros(len(data), dtype=bool)),
            data.non_tensor_batch.get("rollout_timed_out", np.zeros(len(data), dtype=bool)),
        ),
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["trajectory_loss_weights"] = loss_weights
    data.meta_info["compaction_trajectory_count"] = trajectory_count
    return advantages, returns


def prepare_recap_advantages(batch: DataProto, trainer, compute_log_prob) -> tuple[DataProto, dict[str, float]]:
    """Choose the RECAP scoring policy and attach fixed auxiliary advantages."""
    # 1. Validate the supported training configuration.
    config = trainer.config.algorithm.get("compaction_distortion", {})
    if not config.get("enabled", False):
        return batch, {}
    recap_coefficients(config)
    if trainer.config.algorithm.get("adv_estimator") != AdvantageEstimator.GRPO:
        raise ValueError("RECAP requires adv_estimator='grpo'.")
    if trainer.config.actor_rollout_ref.actor.loss_agg_mode != "trajectory-mean-token-mean":
        raise ValueError("RECAP requires loss_agg_mode='trajectory-mean-token-mean'.")
    correction = trainer.config.algorithm.get("rollout_correction", {})
    if not correction or not correction.get("bypass_mode", False):
        raise ValueError("RECAP requires rollout_correction.bypass_mode=true.")

    # 2. EMA replaces the full-context teacher. The scoring policy selects the student.
    scores = None
    scoring_policy = config.get("scoring_policy", "pi_rollout")
    ema = config.get("teacher_ema_rate") is not None
    if scoring_policy not in {"pi_rollout", "pi_fsdp"}:
        raise ValueError("RECAP scoring_policy must be 'pi_rollout' or 'pi_fsdp'.")
    if scoring_policy == "pi_fsdp" or ema:
        rollout = trainer.config.actor_rollout_ref.rollout
        max_model_len = rollout.get("max_model_len") or (rollout.prompt_length + rollout.response_length)
        scores = score_recap_with_trainer(
            batch,
            compute_log_prob,
            pad_token_id=trainer.tokenizer.pad_token_id,
            max_model_len=max_model_len,
            dp_size=trainer._get_dp_size(trainer.actor_rollout_wg, "actor"),
            ema_teacher=ema,
            rollout_student=scoring_policy == "pi_rollout",
        )

    # 3. Attach fixed auxiliary advantages and report the same event-level diagnostics.
    distortions, version_gaps, eligible_events, total_events = compute_recap_advantages(
        batch,
        summary_baseline=config.get("summary_baseline", "leave_one_out"),
        sd_advantage_clip=config.get("sd_advantage_clip", 5.0),
        scores=scores,
    )
    metrics = {}
    if total_events:
        metrics["compaction/summary_pg_eligible_fraction"] = eligible_events / total_events
    if len(distortions):
        metrics["compaction/mean_distortion"] = distortions.mean().item()
    if version_gaps:
        metrics["compaction/mean_teacher_rollout_version_gap"] = float(np.mean(version_gaps))
    return batch, metrics


def _compaction_pairs(batch: DataProto) -> list[tuple[int, int]]:
    """Match summaries to their next action segment, ignoring padding and terminal summaries."""
    padding = batch.non_tensor_batch.get("compaction_padding", np.zeros(len(batch), dtype=bool))
    trajectory_ids = batch.non_tensor_batch["compaction_trajectory_uid"]
    indices = batch.non_tensor_batch["compaction_window_index"]
    rows = {(trajectory_ids[row], int(indices[row])): row for row in range(len(batch)) if not padding[row]}
    pairs = []
    for row in range(len(batch)):
        if padding[row] or not batch.batch["compaction_summary_mask"][row].any():
            continue
        continuation = rows.get((trajectory_ids[row], int(indices[row]) + 1))
        if continuation is None:
            counts = batch.non_tensor_batch.get("compaction_window_count")
            if counts is not None and int(indices[row]) + 1 == int(counts[row]):
                continue
            raise ValueError("A RECAP summary row is missing its following continuation segment.")
        pairs.append((row, continuation))
    return pairs


def score_recap_with_trainer(
    batch: DataProto,
    compute_log_prob,
    *,
    pad_token_id: int,
    max_model_len: int,
    dp_size: int,
    ema_teacher: bool = False,
    rollout_student: bool = False,
):
    """Align full-context teacher and compressed-context student scores to rollout tokens."""
    teacher = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    student = torch.zeros_like(teacher)
    score_mask = torch.zeros_like(batch.batch["response_mask"])
    prompts, responses, masks, destinations = [], [], [], []
    prompt_width = batch.batch["prompts"].shape[1]

    # Keep exactly the rollout teacher's prefix limit and exclude replayed turns.
    for summary_row, row in _compaction_pairs(batch):
        full_context = batch.non_tensor_batch["compaction_pre_context_ids"][summary_row]
        if full_context is None or len(full_context) == 0:
            raise ValueError("Trainer RECAP scoring requires the original pre-compaction context.")
        start = int(batch.non_tensor_batch["compaction_continuation_start"][row])
        action_mask = batch.batch["compaction_continuation_action_mask"][row] * batch.batch["response_mask"][row]
        capacity = max(0, max_model_len - len(full_context) - 1)
        positions = action_mask[start : start + capacity].nonzero().flatten()
        if not len(positions):
            continue
        length = int(positions[-1]) + 1
        continuation = batch.batch["responses"][row, start : start + length].tolist()
        compact_context = batch.batch["prompts"][row][batch.batch["attention_mask"][row, :prompt_width].bool()].tolist()
        compact_context += batch.batch["responses"][row, :start].tolist()
        contexts = [list(full_context)] if rollout_student else [list(full_context), compact_context]
        for context in contexts:
            prompts.append(context)
            responses.append(continuation)
            masks.append(action_mask[start : start + length])
        destinations.append((row, start, length))

    if not destinations:
        return teacher, student, score_mask

    # Use the existing trainer scoring interface. Padding only serves DP dispatch.
    device = batch.batch["responses"].device
    prompt_length = max(map(len, prompts))
    response_length = max(map(len, responses))
    prompt_ids = torch.full((len(prompts), prompt_length), pad_token_id, dtype=torch.long, device=device)
    response_ids = torch.full((len(prompts), response_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(prompts), prompt_length + response_length), dtype=torch.long, device=device)
    response_mask = torch.zeros_like(response_ids)
    for row, (context, continuation, mask) in enumerate(zip(prompts, responses, masks, strict=True)):
        prompt_ids[row, -len(context) :] = torch.tensor(context, device=device)
        response_ids[row, : len(continuation)] = torch.tensor(continuation, device=device)
        attention_mask[row, prompt_length - len(context) : prompt_length + len(continuation)] = 1
        response_mask[row, : len(continuation)] = mask
    scoring_batch = DataProto.from_dict(
        tensors={
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": torch.cat((prompt_ids, response_ids), dim=-1),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
        },
        meta_info={"temperature": 1.0},
    )

    @torch.no_grad()
    def score(rows, use_ema=False):
        padded, _ = pad_dataproto_to_divisor(rows, dp_size)
        padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(-1).tolist()
        scored, _ = compute_log_prob(padded, use_ema=True) if use_ema else compute_log_prob(padded)
        return scored.batch["old_log_probs"][: len(rows)].detach()

    if rollout_student:
        teacher_scores = score(scoring_batch, use_ema=True)
        student = batch.batch["rollout_log_probs"]
    elif ema_teacher:
        teacher_scores = score(scoring_batch[::2], use_ema=True)
        student_scores = score(scoring_batch[1::2])
    else:
        paired_scores = score(scoring_batch)
        teacher_scores, student_scores = paired_scores[::2], paired_scores[1::2]

    for index, (row, start, length) in enumerate(destinations):
        teacher[row, start : start + length] = teacher_scores[index, :length]
        if not rollout_student:
            student[row, start : start + length] = student_scores[index, :length]
        score_mask[row, start : start + length] = masks[index if rollout_student else 2 * index]
    return teacher, student, score_mask


def compute_recap_advantages(
    batch: DataProto, *, summary_baseline: str = "leave_one_out", sd_advantage_clip: float | None = 5.0, scores=None
) -> tuple[torch.Tensor, list[int], int, int]:
    """Build detached summary/continuation advantages from aligned full/compact scores."""
    required_batch_keys = {
        "compaction_summary_mask",
        "compaction_continuation_action_mask",
        "response_mask",
    }
    if scores is None:
        required_batch_keys.update(
            {"compaction_teacher_rollout_logprobs", "compaction_teacher_rollout_mask", "rollout_log_probs"}
        )
    required_non_tensor_keys = {"compaction_trajectory_uid", "compaction_window_index"}
    if not required_batch_keys.issubset(batch.batch.keys()) or not required_non_tensor_keys.issubset(
        batch.non_tensor_batch
    ):
        raise ValueError("RECAP requires complete compaction metadata and scores.")
    if summary_baseline not in {"leave_one_out", "none"}:
        raise ValueError("RECAP summary_baseline must be 'leave_one_out' or 'none'.")
    if sd_advantage_clip is not None and sd_advantage_clip <= 0:
        raise ValueError("RECAP sd_advantage_clip must be positive or null.")

    response_mask = batch.batch["response_mask"].to(torch.bool)
    summary_mask = batch.batch["compaction_summary_mask"].to(torch.bool)
    continuation_mask = batch.batch["compaction_continuation_action_mask"].to(torch.bool)
    if scores is None:
        teacher_logprobs = batch.batch["compaction_teacher_rollout_logprobs"]
        student_logprobs = batch.batch["rollout_log_probs"]
        teacher_mask = batch.batch["compaction_teacher_rollout_mask"]
    else:
        teacher_logprobs, student_logprobs, teacher_mask = scores
    student_logprobs = student_logprobs.to(teacher_logprobs)
    teacher_mask = teacher_mask.to(torch.bool)

    summary_advantages = torch.zeros_like(teacher_logprobs)
    continuation_advantages = torch.zeros_like(teacher_logprobs)
    distortions = torch.zeros(len(batch), dtype=teacher_logprobs.dtype, device=teacher_logprobs.device)
    event_mask = torch.zeros(len(batch), dtype=torch.bool, device=teacher_logprobs.device)
    version_gaps: list[int] = []
    cached_version_gaps = (
        batch.non_tensor_batch.get("compaction_teacher_rollout_version_gap") if scores is None else None
    )

    for summary_row, continuation_row in _compaction_pairs(batch):
        if torch.any(teacher_mask[continuation_row] & ~continuation_mask[continuation_row]):
            raise ValueError("The RECAP teacher mask selects a non-continuation token.")
        scored_mask = (
            teacher_mask[continuation_row] & continuation_mask[continuation_row] & response_mask[continuation_row]
        )
        if not scored_mask.any():
            continue

        current_summary_mask = summary_mask[summary_row] & response_mask[summary_row]
        if not current_summary_mask.any():
            raise ValueError("Every scored RECAP event must contain a trainable summary token.")

        # Both contexts score the same realized continuation. These advantages
        # stay fixed throughout PPO updates, independently of the ratio denominator.
        raw_post_advantage = (
            teacher_logprobs[continuation_row, scored_mask] - student_logprobs[continuation_row, scored_mask]
        ).detach()
        token_count = len(raw_post_advantage)
        # Summary PG uses the raw mean; only continuation SD clips token differences.
        summary_reward = raw_post_advantage.mean()
        distortions[summary_row] = -summary_reward
        summary_advantages[summary_row, current_summary_mask] = summary_reward
        sd = raw_post_advantage
        if sd_advantage_clip is not None:
            sd = sd.clamp(-sd_advantage_clip, sd_advantage_clip)
        continuation_advantages[continuation_row, scored_mask] += sd / token_count
        event_mask[summary_row] = True

        if cached_version_gaps is not None:
            version_gap = cached_version_gaps[continuation_row]
            if version_gap is not None:
                version_gaps.append(int(version_gap))

    batch.batch["compaction_distortions"] = distortions
    batch.batch["compaction_distortion_event_mask"] = event_mask
    batch.batch["compaction_continuation_advantages"] = continuation_advantages
    total_events = int(event_mask.sum())
    eligible_events = total_events
    if summary_baseline == "leave_one_out":
        summary_advantages, eligible_events, total_events = _summary_leave_one_out(batch, summary_advantages)
    batch.batch["compaction_summary_advantages"] = summary_advantages
    return distortions[event_mask], version_gaps, eligible_events, total_events


def _summary_leave_one_out(
    batch: DataProto,
    raw_summary_advantages: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Center summary rewards within each task and compaction index; skip singleton groups."""
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("Leave-one-out compaction PG requires GRPO prompt-group identifiers in 'uid'.")

    response_mask = batch.batch["response_mask"].to(torch.bool)
    summary_mask = batch.batch["compaction_summary_mask"].to(torch.bool) & response_mask
    event_mask = batch.batch["compaction_distortion_event_mask"].to(torch.bool)
    total_event_count = int(event_mask.sum().item())

    trajectory_ids = batch.non_tensor_batch["compaction_trajectory_uid"]
    window_indices = batch.non_tensor_batch["compaction_window_index"]
    prompt_group_ids = batch.non_tensor_batch["uid"]
    padding = np.asarray(
        batch.non_tensor_batch.get("compaction_padding", np.zeros(len(batch), dtype=bool)),
        dtype=bool,
    )

    # Compaction index means the event's ordinal inside its logical trajectory,
    # not its flattened batch row. This aligns the first, second, ... events
    # even when different rollouts contain different numbers of compactions.
    rows_by_trajectory: dict[object, list[int]] = defaultdict(list)
    for row in range(len(batch)):
        if not padding[row] and summary_mask[row].any():
            rows_by_trajectory[trajectory_ids[row]].append(row)

    rows_by_group_and_index: dict[tuple[object, int], list[int]] = defaultdict(list)
    for rows in rows_by_trajectory.values():
        rows.sort(key=lambda row: int(window_indices[row]))
        for compaction_index, row in enumerate(rows):
            if not event_mask[row]:
                continue
            prompt_group_id = prompt_group_ids[row]
            if isinstance(prompt_group_id, np.generic):
                prompt_group_id = prompt_group_id.item()
            rows_by_group_and_index[(prompt_group_id, compaction_index)].append(row)

    pg_advantages = torch.zeros_like(raw_summary_advantages)
    eligible_event_count = 0

    for rows in rows_by_group_and_index.values():
        # A singleton has no unbiased peer baseline. Mask only this event while
        # retaining its task and self-distillation objectives.
        if len(rows) < 2:
            continue

        rewards = []
        for row in rows:
            current_summary_mask = summary_mask[row]
            if not current_summary_mask.any():
                raise ValueError("Every scored RECAP event must contain a trainable summary token.")
            rewards.append(raw_summary_advantages[row, current_summary_mask][0])

        reward_sum = torch.stack(rewards).sum()
        for row, reward in zip(rows, rewards, strict=True):
            leave_one_out_mean = (reward_sum - reward) / (len(rows) - 1)
            pg_advantages[row, summary_mask[row]] = reward - leave_one_out_mean
            eligible_event_count += 1

    return pg_advantages, eligible_event_count, total_event_count


def apply_recap_advantage(batch: DataProto, config) -> DataProto:
    """Apply configured weights after task GRPO; keep joint/separate clipping unchanged."""
    if not config.get("enabled", False):
        return batch
    summary_coef, continuation_coef = recap_coefficients(config)
    clipping = config.get("clipping", "joint")
    if clipping not in {"joint", "separate"}:
        raise ValueError("RECAP clipping must be 'joint' or 'separate'.")

    summary_advantages = batch.batch.get("compaction_summary_advantages")
    continuation_advantages = batch.batch.get("compaction_continuation_advantages")
    if summary_advantages is None and continuation_advantages is None:
        return batch
    if summary_advantages is None or continuation_advantages is None:
        raise ValueError("RECAP requires both compaction advantage tensors.")
    if "trajectory_loss_weights" not in batch.batch:
        raise ValueError("RECAP requires trajectory-mean-token-mean loss weights.")

    if clipping == "joint":
        auxiliary_advantage = summary_coef * summary_advantages + continuation_coef * continuation_advantages
        batch.batch["advantages"] = batch.batch["advantages"] + auxiliary_advantage.to(batch.batch["advantages"])
    else:
        # Coefficients are applied once here; all terms retain the same loss normalization.
        if summary_coef:
            batch.batch["recap_summary_loss_advantages"] = summary_coef * summary_advantages
        if continuation_coef:
            batch.batch["recap_continuation_loss_advantages"] = continuation_coef * continuation_advantages
    return batch
