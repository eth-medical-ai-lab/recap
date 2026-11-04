"""Score a post-compaction continuation under its original full context."""

from __future__ import annotations


async def score_rollout_teacher(
    server_manager,
    *,
    request_id: str,
    max_model_len: int,
    pre_compaction_context_ids: list[int],
    continuation_ids: list[int],
    continuation_action_mask: list[int],
    student_min_global_step: int | None,
    student_max_global_step: int | None,
) -> tuple[list[float], list[int], int | None]:
    """Return continuation-aligned logprobs, an action mask, and the version gap.

    Score the longest prefix that fits and ends in an action token. Observations
    remain in the teacher context, but only action tokens are selected for RECAP.
    Unscored positions stay zero. This function does not modify segment state.
    """
    # 1. Select the continuation prefix that fits beside the original context.
    if len(continuation_ids) != len(continuation_action_mask):
        raise ValueError("A compaction continuation has misaligned tokens and action mask.")
    if not pre_compaction_context_ids:
        raise ValueError("Rollout-teacher scoring requires a non-empty pre-compaction context.")
    teacher_logprobs = [0.0] * len(continuation_ids)
    teacher_mask = [0] * len(continuation_ids)
    # vLLM prompt scoring still generates one ignored token; reserve space for it.
    capacity = max(0, max_model_len - len(pre_compaction_context_ids) - 1)
    action_positions = [index for index, selected in enumerate(continuation_action_mask[:capacity]) if selected]
    if not action_positions:
        return teacher_logprobs, teacher_mask, None
    scored_length = action_positions[-1] + 1
    scored_continuation = continuation_ids[:scored_length]

    # 2. Score the realized tokens with the rollout policy under the full context.
    output = await server_manager.generate(
        request_id=request_id,
        prompt_ids=pre_compaction_context_ids + scored_continuation,
        sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
    )

    # 3. Align vLLM's shifted prompt scores with the continuation token IDs.
    # Entry i scores token i + 1; the final entry is a dummy generated-token score.
    fields = output.extra_fields
    prompt_ids = fields.get("prompt_ids")
    prompt_logprobs = fields.get("prompt_logprobs")
    if prompt_ids is None or prompt_logprobs is None:
        raise ValueError("The rollout server did not return prompt log probabilities for the RECAP teacher.")
    start = len(pre_compaction_context_ids) - 1
    end = start + scored_length
    scored_ids = [int(item[0]) for item in prompt_ids[start:end]]
    scored_logprobs = [float(item[0]) for item in prompt_logprobs[start:end]]
    if scored_ids != scored_continuation or len(scored_logprobs) != scored_length:
        raise ValueError("Rollout-teacher prompt log probabilities are not aligned with the continuation.")
    teacher_logprobs[:scored_length] = scored_logprobs
    teacher_mask[:scored_length] = continuation_action_mask[:scored_length]

    # 4. Compare all student/teacher version endpoints; absent metadata stays unknown.
    teacher_step = fields.get("global_steps")
    teacher_min_step = fields.get("min_global_steps", teacher_step)
    teacher_max_step = fields.get("max_global_steps", teacher_step)
    student_steps = [step for step in (student_min_global_step, student_max_global_step) if step is not None]
    teacher_steps = [step for step in (teacher_min_step, teacher_max_step) if step is not None]
    version_gap = None
    if student_steps and teacher_steps:
        version_gap = max(abs(int(student) - int(teacher)) for student in student_steps for teacher in teacher_steps)
    return teacher_logprobs, teacher_mask, version_gap
