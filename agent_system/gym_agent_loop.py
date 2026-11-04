"""Gym agent loop with action and summary segments for RECAP."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agent_system.compaction import build_compacted_messages, flatten_retained_turn_tokens, get_compaction_instruction
from agent_system.env_base import AgentEnv, TaskId
from agent_system.env_registry import make_env
from agent_system.envs.swe_agent import SweAgentServiceError
from agent_system.recap_teacher import score_rollout_teacher
from agent_system.trajectory_logging import (
    trajectory_directory,
    write_swebench_failure,
    write_swebench_solution,
    write_trajectory_txt,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.trainer.ppo.recap import recap_needs_scoring
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """One prompt and its continuation; masks and logprobs align with response_ids.

    Action segments can contain many model/tool turns. Environment tokens belong
    to the response too, but only generated tokens have response_mask=1.
    """

    kind: Literal["action", "summary"]
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = field(default_factory=list)
    routed_experts: Any = None
    summary_text: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)  # Only populated when saving trajectories.
    min_global_step: int | None = None
    max_global_step: int | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)

    # A summary keeps its original context; the following action owns the scores.
    pre_compaction_context_ids: list[int] | None = None
    continuation_start: int = 0  # Response offset after any replayed turns.
    teacher_logprobs: list[float] | None = None
    teacher_mask: list[int] | None = None
    teacher_version_gap: int | None = None

    def record_generation(self, output) -> None:
        """Record policy versions and preemptions for this segment's generation."""
        fields = output.extra_fields
        step = fields.get("global_steps")
        minimum = fields.get("min_global_steps", step)
        maximum = fields.get("max_global_steps", step)
        if minimum is not None:
            self.min_global_step = (
                int(minimum) if self.min_global_step is None else min(self.min_global_step, int(minimum))
            )
        if maximum is not None:
            self.max_global_step = (
                int(maximum) if self.max_global_step is None else max(self.max_global_step, int(maximum))
            )
        if output.num_preempted is not None:
            self.metrics["num_preempted"] = self.metrics.get("num_preempted", 0) + output.num_preempted


class GymAgentLoop(AgentLoopBase):
    """Run an episode as an ordered sequence of training segments.

    Without compaction: one episode = action segment 1.
    With compaction: one episode = action segment 1 + summary segment 2
        + action segment 3 + summary segment 4 + ...

    An action segment can contain multiple model/environment turns. A summary
    segment generates a handoff; the next action segment resumes the same env.
    All segments share the episode's final task reward. If the episode budget
    runs out during a summary, there is no subsequent action segment.
    """

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs):
        if not data_config.config.continuous_token.enable:
            raise ValueError("The segment-based loop requires data.continuous_token.enable=True.")
        # Both environments are text-only. Use the tokenizer's continuous-token
        # builder even when Qwen3.5 also loads a processor.
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )
        self.processor = processor
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_len = self.rollout_config.get("max_model_len") or (
            self.rollout_config.prompt_length + self.response_length
        )
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 12
        self.max_tokens = self.config.trainer.get("gym_max_tokens")
        if self.max_tokens is not None:
            self.max_tokens = int(self.max_tokens)
            if self.max_tokens <= 0:
                raise ValueError("trainer.gym_max_tokens must be positive or null.")
        self.tool_parser = ToolParser.get_tool_parser(str(self.rollout_config.multi_turn.format), tokenizer)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput | list[AgentLoopOutput]:
        """Resolve a dataset row and convert each episode segment to a training sample."""
        # 1. Resolve the task and environment configuration from the dataset row.
        trainer = self.config.trainer
        compact = bool(trainer.get("gym_compaction_enabled", False))
        extra_info = kwargs.get("extra_info") or {}
        env_name = kwargs.get("env_name") or extra_info.get("env_name")
        if env_name is None:
            raise ValueError("GymAgentLoop requires env_name in the dataset row or extra_info.")
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = extra_info.get("task_id")
        split = kwargs.get("split") or extra_info.get("split") or "train"
        env_kwargs = dict(trainer.get("gym_env_kwargs") or {})
        env_kwargs.update(kwargs.get("env_kwargs") or extra_info.get("env_kwargs") or {})

        # 2. Set explicit token and wall-clock budgets for the whole episode.
        # Both environments use the explicit setting; None adds no episode-wide cap.
        episode_token_budget = trainer.get("gym_max_episode_tokens")
        if episode_token_budget is not None:
            episode_token_budget = int(episode_token_budget)
            if episode_token_budget <= 0:
                raise ValueError("trainer.gym_max_episode_tokens must be positive or null.")
        timeout_seconds = trainer.get("gym_rollout_timeout_seconds")
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds <= 0:
                raise ValueError("trainer.gym_rollout_timeout_seconds must be positive or null.")

        # 3. Cache rollout scores, or retain contexts for trainer-side RECAP scoring.
        distortion = self.config.algorithm.get("compaction_distortion", {})
        needs_scores = compact and not kwargs.get("validate", False) and recap_needs_scoring(distortion)
        scoring_policy = distortion.get("scoring_policy", "pi_rollout")
        if needs_scores and scoring_policy not in {"pi_rollout", "pi_fsdp"}:
            raise ValueError("RECAP scoring_policy must be 'pi_rollout' or 'pi_fsdp'.")
        cache_teacher = needs_scores and scoring_policy == "pi_rollout" and distortion.get("teacher_ema_rate") is None
        retain_teacher_context = needs_scores and not cache_teacher
        # Score the realized continuation under the unfiltered training distribution.
        if needs_scores and (
            float(sampling_params.get("temperature", 1.0)) != 1.0
            or float(sampling_params.get("top_p", 1.0)) != 1.0
            or int(sampling_params.get("top_k", -1)) not in {-1, 0}
            or float(sampling_params.get("repetition_penalty", 1.0)) != 1.0
        ):
            raise ValueError(
                "RECAP requires unfiltered training sampling: temperature=1.0, top_p=1.0, "
                "top_k=-1 (or 0), and repetition_penalty=1.0."
            )

        # 4. Run the episode once. Service-operation retries belong to SWE itself.
        trajectory_dir = trajectory_directory(
            trainer, kwargs.get("validate", False), kwargs.get("log_trajectory_txt", False)
        )
        env = make_env(env_name, split=split, **env_kwargs)
        started_at = time.monotonic()
        deadline = asyncio.timeout(timeout_seconds)
        try:
            async with deadline:
                segments, outcome = await self._run_episode(
                    sampling_params,
                    env=env,
                    task_id=task_id,
                    split=split,
                    episode_token_budget=episode_token_budget,
                    cache_teacher=cache_teacher,
                    retain_teacher_context=retain_teacher_context,
                    log_trajectory=trajectory_dir is not None,
                )
        except TimeoutError:
            # Only our deadline is an episode timeout; model/service timeouts still raise.
            if not deadline.expired():
                raise
            logger.warning("Episode timed out after %ss: env=%s task=%s", timeout_seconds, env_name, task_id)
            segments = []
            outcome = {
                "reward_score": 0.0,
                "done": False,
                "finish_reason": "rollout_timeout",
                "num_turns": 0,
                "trajectory_id": uuid4().hex,
                "episode_token_count": 0,
                "compaction_count": 0,
                "reset_info": {},
                "step_infos": [],
                "failure": {"kind": "rollout_timeout", "message": f"Episode exceeded {timeout_seconds:g}s."},
                "cleanup_failed": False,
                "assistant_turns": 0,
                "metrics": {},
            }

        # Reset failures and timeouts keep a zero-reward slot with no training tokens.
        if not segments:
            segments = [Segment(kind="action", prompt_ids=[])]
            if trajectory_dir is not None and (failure := outcome["failure"]):
                segments[0].messages.append({"role": "error", "content": failure["message"]})

        # 5. Prepare the final reward and episode metadata shared by all segments.
        raw_reward = outcome["reward_score"]
        reward_score = float(raw_reward >= 1.0)
        final_info = outcome["step_infos"][-1] if outcome["step_infos"] else {}
        # Count new activity, excluding history copied into summaries or replayed turns.
        summary_lengths = [sum(segment.response_mask) for segment in segments if segment.kind == "summary"]
        format_checks = [bool(info["format_ok"]) for info in outcome["step_infos"] if "format_ok" in info]
        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "env_name": env_name,
            "task_id": outcome["reset_info"].get("task_id", task_id),
            "split": split,
            "env_reset_info": outcome["reset_info"],
            "env_step_infos": outcome["step_infos"],
            "final_observation_messages": final_info.get("final_observation_messages", []),
            "finish_reason": outcome["finish_reason"],
            "raw_reward_score": raw_reward,
            "task_reward_score": reward_score,
            "episode_reward": raw_reward,
            "episode_success": reward_score,
            "episode_length": outcome["assistant_turns"],
            "episode_tool_call_count": sum(len(info.get("tool_calls") or []) for info in outcome["step_infos"]),
            "episode_response_length": outcome["episode_token_count"],
            "episode_action_token_count": sum(
                sum(segment.response_mask) for segment in segments if segment.kind == "action"
            ),
            "episode_summary_token_count": sum(summary_lengths),
            "episode_done": float(outcome["done"]),
            "episode_terminated": float(final_info.get("terminated", final_info.get("completed", False))),
            "episode_truncated": float(bool(outcome["failure"]) or final_info.get("truncated", not outcome["done"])),
            "episode_answer_submitted": float(final_info.get("finish_count", 0) > 0)
            if env_name.startswith("browsecomp_plus")
            else None,
            "episode_token_budget": episode_token_budget,
            "episode_token_count": outcome["episode_token_count"],
            "episode_token_limit_hit": float(outcome["finish_reason"] == "episode_token_budget"),
            "format_ok_rate": sum(format_checks) / len(format_checks) if format_checks else 0.0,
            "invalid_format_count": len(format_checks) - sum(format_checks),
            "compaction_enabled": compact,
            "compaction_count": outcome["compaction_count"],
            "compaction_trajectory_uid": outcome["trajectory_id"],
            "rollout_elapsed_seconds": time.monotonic() - started_at,
            "rollout_timeout_seconds": timeout_seconds,
            "rollout_timed_out": outcome["finish_reason"] == "rollout_timeout",
            "rollout_failure_type": outcome["failure"]["kind"] if outcome["failure"] else None,
            "swe_infrastructure_failure": bool(outcome["failure"] and outcome["failure"].get("infrastructure", False)),
            "swe_cleanup_failed": outcome["cleanup_failed"],
        }

        # 6. Save diagnostics separately from the training tensors.
        filename = outcome["trajectory_id"]
        metadata = {
            **extra_fields,
            "trajectory_id": filename,
            "num_turns": outcome["num_turns"],
            "failure": outcome["failure"],
            "global_steps": kwargs.get("global_steps"),
            "sample_index": kwargs.get("sample_index"),
            "rollout_n": kwargs.get("rollout_n"),
        }
        # Messages belong in the transcript, not repeated in every segment's header.
        for key in ("env_reset_info", "env_step_infos", "final_observation_messages"):
            metadata.pop(key)
        if trajectory_dir is not None:
            try:
                paths = write_trajectory_txt(trajectory_dir / str(split) / filename, metadata, segments)
                extra_fields["gym_trajectory_txt_paths"] = [str(path) for path in paths]
                extra_fields["gym_trajectory_txt_path"] = str(paths[0] if len(paths) == 1 else paths[0].parent)
            except OSError:
                logger.warning("Could not save trajectory task=%s", task_id, exc_info=True)
        solution = final_info.get("swebench_solution")
        solution_dir = trainer.get("gym_swebench_solution_dir")
        if solution is not None and solution_dir:
            path = Path(solution_dir) / str(split) / f"{filename}.json"
            try:
                write_swebench_solution(path, solution)
                extra_fields["gym_swebench_solution_path"] = str(path)
            except OSError:
                logger.warning("Could not save SWE solution task=%s", task_id, exc_info=True)
        failure_dir = trainer.get("gym_swebench_failure_dir")
        if outcome["failure"] and env_name.startswith("swe_agent") and failure_dir:
            path = Path(failure_dir) / str(split) / f"{filename}.json"
            try:
                write_swebench_failure(path, metadata)
                extra_fields["gym_swebench_failure_path"] = str(path)
            except OSError:
                logger.warning("Could not save SWE failure task=%s", task_id, exc_info=True)

        # 7. Convert each segment into a training output with its own token masks.
        # Keep verl's existing window field names only at the training interface.
        outputs = []
        metrics_index = len(segments) - 1 if kwargs.get("validate", False) else 0
        for index, segment in enumerate(segments):
            # Scalar episode fields are deduplicated by trajectory ID downstream.
            # Per-summary lengths remain for evaluation; environment time appears once.
            segment_fields = {
                "min_global_steps": segment.min_global_step,
                "max_global_steps": segment.max_global_step,
                "compact_response_tokens": summary_lengths if index == metrics_index else None,
            }
            if compact:
                segment_fields.update(
                    {
                        "segment_kind": segment.kind,
                        "compaction_window_index": index,
                        "compaction_window_count": len(segments),
                        "compaction_summary_token_count": sum(segment.response_mask)
                        if segment.kind == "summary"
                        else 0,
                        "compaction_summary_valid": float(bool(segment.summary_text)),
                        "compaction_teacher_rollout_version_gap": segment.teacher_version_gap,
                    }
                )
            teacher_logprobs, teacher_mask = segment.teacher_logprobs, segment.teacher_mask
            if retain_teacher_context:
                segment_fields["compaction_pre_context_ids"] = segment.pre_compaction_context_ids
                segment_fields["compaction_continuation_start"] = segment.continuation_start
            if cache_teacher and teacher_logprobs is None:
                # Uniform tensor fields are required even for rows without a continuation.
                teacher_logprobs = [0.0] * len(segment.response_ids)
                teacher_mask = [0] * len(segment.response_ids)
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=segment.prompt_ids,
                    response_ids=segment.response_ids,
                    response_mask=segment.response_mask,
                    response_logprobs=segment.response_logprobs or None,
                    compaction_teacher_rollout_logprobs=teacher_logprobs,
                    compaction_teacher_rollout_mask=teacher_mask,
                    compaction_summary_mask=(
                        segment.response_mask if segment.kind == "summary" else [0] * len(segment.response_ids)
                    )
                    if compact
                    else None,
                    compaction_continuation_action_mask=(
                        segment.response_mask
                        if segment.kind == "action" and index > 0
                        else [0] * len(segment.response_ids)
                    )
                    if compact
                    else None,
                    routed_experts=(
                        segment.routed_experts[: len(segment.prompt_ids) + self.response_length]
                        if segment.routed_experts is not None
                        else None
                    ),
                    reward_score=reward_score,
                    num_turns=outcome["num_turns"],
                    metrics={**segment.metrics, **(outcome["metrics"] if index == metrics_index else {})},
                    extra_fields={**extra_fields, **segment_fields},
                )
            )

        # 8. Select the outputs for training or validation.
        if compact:
            # Validation counts each task once, regardless of how often it compacted.
            return [outputs[-1]] if kwargs.get("validate", False) else outputs
        return outputs[0]

    async def _run_episode(
        self,
        sampling_params: dict[str, Any],
        *,
        env: AgentEnv,
        task_id: TaskId | None = None,
        split: str = "train",
        episode_token_budget: int | None = None,
        cache_teacher: bool = False,
        retain_teacher_context: bool = False,
        log_trajectory: bool = False,
    ) -> tuple[list[Segment], dict[str, Any]]:
        """Run one episode and return its segments and outcome.

        The caller supplies an async BrowseComp-Plus or SWE environment. This
        method owns reset/close. SWE evaluates its current patch on truncation.
        """
        # Initialize episode state.
        request_id = uuid4().hex
        trajectory_id = request_id
        reward_score, done = 0.0, False
        assistant_turns = 0
        user_turns = 0
        finish_reason = "max_assistant_turns"
        step_infos = []
        segments, reset_info = [], {}
        failure = None
        cleanup_failed = False
        metrics = {}
        episode_token_count = 0
        compaction_count = 0
        recent_turn_token_ids, recent_turn_messages = [], []
        latest_summary = None
        trainer = self.config.trainer
        compact = bool(trainer.get("gym_compaction_enabled", False))
        context_threshold = int(trainer.get("gym_compaction_context_threshold", 8192))
        max_compactions = int(trainer.get("gym_compaction_max_compactions", 5))
        keep_recent_turns = int(trainer.get("gym_compaction_keep_recent_turns", 1))

        try:
            # Reset the environment and start the first action segment.
            with simple_timer("tool_calls", metrics):
                messages, reset_info = await env.reset(task_id=task_id, split=split)
            user_turns = 1
            messages = list(messages)
            initial_messages = list(messages)
            env_name = reset_info["env_name"]
            tool_schema_dicts = reset_info.get("tool_schemas") or []
            tool_schemas = [OpenAIFunctionToolSchema.model_validate(schema) for schema in tool_schema_dicts]
            prompt_ids = await self.ct_build_initial_tokens(
                messages, tools=tool_schema_dicts or None, truncate_to_prompt_length=not compact
            )
            if compact and len(prompt_ids) > self.prompt_length:
                raise ValueError("The original task exceeds rollout.prompt_length; compaction must preserve it.")
            segment = Segment(kind="action", prompt_ids=prompt_ids)
            segments = [segment]
            if log_trajectory:
                segment.messages = (
                    [{"role": "tools", "content": tool_schema_dicts}] if tool_schema_dicts else []
                ) + messages

            for _ in range(self.max_assistant_turns):
                # 1. Check token budgets before this turn.
                # The live model context is exactly the segment's prompt + response.
                token_ids = segment.prompt_ids + segment.response_ids
                context_budget = self.max_model_len - len(segment.prompt_ids)
                response_budget = min(self.response_length, context_budget)
                if episode_token_budget is not None:
                    response_budget = min(
                        response_budget, len(segment.response_ids) + episode_token_budget - episode_token_count
                    )
                remaining_tokens = response_budget - len(segment.response_ids)
                if remaining_tokens <= 0:
                    finish_reason = (
                        "response_length" if len(segment.response_ids) >= self.response_length else "context_length"
                    )
                    if episode_token_budget is not None and episode_token_count >= episode_token_budget:
                        finish_reason = "episode_token_budget"
                    break
                max_tokens = min(remaining_tokens, self.max_tokens or remaining_tokens)

                # 2. Generate the assistant response and record its training tokens.
                with simple_timer("generate_sequences", segment.metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=token_ids,
                        sampling_params={**sampling_params, "max_tokens": max_tokens},
                    )
                assistant_ids = output.token_ids[:max_tokens]
                episode_token_count += len(assistant_ids)
                segment.response_ids.extend(assistant_ids)
                segment.response_mask.extend([1] * len(assistant_ids))
                if segment.response_logprobs is not None:
                    if not output.log_probs:
                        segment.response_logprobs = None
                    else:
                        segment.response_logprobs.extend(output.log_probs[:max_tokens])
                assistant_turns += 1
                if output.routed_experts is not None:
                    segment.routed_experts = output.routed_experts
                segment.record_generation(output)

                # 3. Execute the gym action and collect its observation and reward.
                # The last generated action still runs even when it fills the budget.
                action, assistant_message = await self._parse_action(
                    assistant_ids, tool_schemas, env_name, assistant_turns
                )
                messages.append(assistant_message)
                if log_trajectory:
                    segment.messages.append(
                        {**assistant_message, "content": self.tokenizer.decode(assistant_ids, skip_special_tokens=True)}
                    )
                with simple_timer("tool_calls", metrics):
                    observations, reward, done, info = await env.step(action)
                if log_trajectory:
                    segment.messages.extend(observations)
                step_infos.append(info)
                if done:
                    reward_score = float(reward)
                    finish_reason = "done"
                    step_infos[-1] = {**info, "final_observation_messages": observations}
                    break

                # 4. Prepare the next turn: check limits, then append the observation.
                # Terminal feedback has already ended the loop and is not encoded.
                if len(output.token_ids) >= max_tokens:
                    finish_reason = (
                        "context_length" if len(segment.response_ids) >= context_budget else "response_length"
                    )
                    if episode_token_budget is not None and episode_token_count >= episode_token_budget:
                        finish_reason = "episode_token_budget"
                    break

                observation_ids = await self._encode_observation(segment, messages, observations, tool_schema_dicts)
                next_response_length = len(segment.response_ids) + len(observation_ids)
                if next_response_length > response_budget:
                    finish_reason = "context_length" if next_response_length > context_budget else "response_length"
                    if (
                        episode_token_budget is not None
                        and episode_token_count + len(observation_ids) > episode_token_budget
                    ):
                        finish_reason = "episode_token_budget"
                    break

                # Preserve sampled tokens verbatim; only new environment context is encoded.
                segment.response_ids.extend(observation_ids)
                segment.response_mask.extend([0] * len(observation_ids))
                if segment.response_logprobs is not None:
                    segment.response_logprobs.extend([0.0] * len(observation_ids))
                messages.extend(observations)
                user_turns += 1
                episode_token_count += len(observation_ids)

                # 5. Compact between complete turns, then resume in a new action segment.
                if compact:
                    recent_turn_token_ids.append(assistant_ids + observation_ids)
                    recent_turn_messages.append([assistant_message, *observations])
                if not (compact and len(segment.prompt_ids) + len(segment.response_ids) >= context_threshold):
                    continue
                if assistant_turns == self.max_assistant_turns:
                    continue
                if episode_token_budget is not None and episode_token_count >= episode_token_budget:
                    finish_reason = "episode_token_budget"
                    break
                if compaction_count >= max_compactions:
                    finish_reason = "max_compactions"
                    break

                # Close the previous continuation before generating another summary.
                if cache_teacher and compaction_count:
                    await self._score_continuation(segment, segments[-2], request_id)
                pre_compaction_context_ids = (
                    segment.prompt_ids + segment.response_ids if cache_teacher or retain_teacher_context else None
                )
                summary_segment, added_tokens, stop_reason = await self._compact(
                    segment,
                    initial_messages,
                    latest_summary,
                    recent_turn_messages,
                    env_name=env_name,
                    sampling_params=sampling_params,
                    episode_tokens_remaining=(
                        episode_token_budget - episode_token_count if episode_token_budget is not None else None
                    ),
                    log_trajectory=log_trajectory,
                )
                if summary_segment is None:
                    finish_reason = stop_reason
                    break
                summary_segment.pre_compaction_context_ids = pre_compaction_context_ids
                segments.append(summary_segment)
                compaction_count += 1
                episode_token_count += added_tokens
                if episode_token_budget is not None and episode_token_count >= episode_token_budget:
                    finish_reason = "episode_token_budget"
                    break

                # Replayed turns are masked context; only newly admitted tokens use budget.
                latest_summary = summary_segment.summary_text or (
                    "No reliable compact state was generated; inspect the retained raw turn."
                )
                retained_ids, retained_count = flatten_retained_turn_tokens(recent_turn_token_ids, keep_recent_turns)
                recent_turn_token_ids = recent_turn_token_ids[-retained_count:] if retained_count else []
                recent_turn_messages = recent_turn_messages[-retained_count:] if retained_count else []
                messages = build_compacted_messages(initial_messages, latest_summary)
                prompt_ids = await self.ct_build_initial_tokens(
                    messages, tools=tool_schema_dicts or None, truncate_to_prompt_length=False
                )
                if len(prompt_ids) > self.prompt_length:
                    raise ValueError("The original task and summary exceed rollout.prompt_length.")
                if (
                    len(retained_ids) >= self.response_length
                    or len(prompt_ids) + len(retained_ids) > self.max_model_len
                ):
                    raise ValueError("The compacted prompt and retained turns exceed the segment capacity.")
                segment = Segment(
                    kind="action",
                    prompt_ids=prompt_ids,
                    response_ids=retained_ids,
                    response_mask=[0] * len(retained_ids),
                    response_logprobs=[0.0] * len(retained_ids),
                    continuation_start=len(retained_ids),
                )
                segments.append(segment)
                messages.extend(message for turn in recent_turn_messages for message in turn)
                if log_trajectory:
                    segment.messages = (
                        [{"role": "tools", "content": tool_schema_dicts}] if tool_schema_dicts else []
                    ) + build_compacted_messages(initial_messages, latest_summary)
                    if retained_ids:
                        segment.messages.append(
                            {
                                "role": "context",
                                "content": self.tokenizer.decode(retained_ids, skip_special_tokens=False),
                            }
                        )
                request_id = uuid4().hex  # The reconstructed context starts a new model request.

            # Finalize at a rollout limit using the environment's scoring rules.
            # Keep the limit reason even though finalization completes the environment.
            if not done:
                with simple_timer("tool_calls", metrics):
                    observations, reward, done, info = await env.truncate(finish_reason)
                if not done:
                    raise RuntimeError("AgentEnv.truncate() must finish the episode.")
                reward_score = float(reward)
                step_infos.append({**info, "final_observation_messages": observations})
                if log_trajectory:
                    segments[-1].messages.extend(observations)
        except SweAgentServiceError as error:
            # The backend classifies the error. Preserve sampled actions, but never replay them.
            if error.is_fatal:
                raise
            logger.warning("Environment failed task=%s kind=%s: %s", task_id, error.kind, error)
            failure = {"kind": error.kind, "message": str(error), "infrastructure": error.is_infrastructure_failure}
            finish_reason = "environment_error" if error.is_infrastructure_failure else error.kind
            reward_score, done = 0.0, False
            step_infos.append({"truncated": True, "terminated": False, "error": str(error)})
            if log_trajectory and segments:
                segments[-1].messages.append({"role": "error", "content": str(error)})
        finally:
            # Cleanup must not replace an evaluated reward or hide the original error.
            try:
                await env.close()
            except Exception:
                cleanup_failed = True
                logger.warning("Environment cleanup failed task=%s", task_id, exc_info=True)

        # Score an open final continuation once, including token/turn-limit stops.
        # A terminal summary has no continuation; a capacity stop may already be scored.
        if cache_teacher and len(segments) > 1 and segments[-1].kind == "action" and segment.teacher_logprobs is None:
            await self._score_continuation(segment, segments[-2], request_id)

        return segments, {
            "reward_score": reward_score,
            "done": done,
            "finish_reason": finish_reason,
            "assistant_turns": assistant_turns,
            "num_turns": user_turns + assistant_turns + 2 * compaction_count,
            "trajectory_id": trajectory_id,
            "episode_token_count": episode_token_count,
            "compaction_count": compaction_count,
            "reset_info": reset_info,
            "step_infos": step_infos,
            "failure": failure,
            "cleanup_failed": cleanup_failed,
            "metrics": metrics,
        }

    async def _score_continuation(self, segment: Segment, summary_segment: Segment, request_id: str):
        """Store teacher scores on the action segment, excluding replayed turns."""
        start = segment.continuation_start
        with simple_timer("compute_score", segment.metrics):
            scores, mask, version_gap = await score_rollout_teacher(
                self.server_manager,
                request_id=request_id,
                max_model_len=self.max_model_len,
                pre_compaction_context_ids=summary_segment.pre_compaction_context_ids,
                continuation_ids=segment.response_ids[start:],
                continuation_action_mask=segment.response_mask[start:],
                student_min_global_step=segment.min_global_step,
                student_max_global_step=segment.max_global_step,
            )
        # Pad back to response coordinates so scores align with the training tensors.
        segment.teacher_logprobs = [0.0] * start + scores
        segment.teacher_mask = [0] * start + mask
        segment.teacher_version_gap = version_gap

    async def _compact(
        self,
        action_segment,
        initial_messages,
        previous_summary,
        recent_turn_messages,
        *,
        env_name,
        sampling_params,
        episode_tokens_remaining,
        log_trajectory=False,
    ) -> tuple[Segment | None, int, str | None]:
        """Close the action segment and generate a summary under a tool-free prompt.

        Copy history tokens exactly, replacing only the trailing assistant prefix
        with the summary instruction. Return the summary, new token cost, and any
        capacity stop reason. The caller reconstructs the next action context.
        """
        # 1. Build the summary's fixed prompt without tool definitions.
        # Re-render only the fixed prefix; prior actions and observations remain exact.
        summary_messages = (
            initial_messages
            if previous_summary is None
            else build_compacted_messages(initial_messages, previous_summary, for_compaction=True)
        )
        prompt_ids = await self.ct_build_initial_tokens(summary_messages, tools=None, truncate_to_prompt_length=False)
        if len(prompt_ids) > self.prompt_length:
            raise ValueError("The tool-free compaction prefix exceeds rollout.prompt_length.")

        # 2. Copy the exact history and encode the summary instruction.
        generation_ids = self.continuous_token_builder.render_delta_token_id(
            initial_messages, [], add_generation_prompt=True
        )
        marker_length = len(generation_ids)
        if not marker_length or action_segment.response_ids[-marker_length:] != generation_ids:
            raise ValueError("Compaction requires a complete turn ending in the assistant generation prefix.")
        history_ids = action_segment.response_ids[:-marker_length]
        summary_segment = Segment(
            kind="summary",
            prompt_ids=prompt_ids,
            response_ids=history_ids,
            response_mask=[0] * len(history_ids),
            response_logprobs=[0.0] * len(history_ids),
        )
        messages = [*summary_messages, *(message for turn in recent_turn_messages for message in turn)]
        instruction = [{"role": "user", "content": get_compaction_instruction(env_name)}]
        instruction_ids = await self._encode_observation(summary_segment, messages, instruction, [])

        # 3. Check summary capacity before modifying the action segment.
        # The instruction replaces an already-counted prefix. Charge only its net addition.
        instruction_cost = max(0, len(instruction_ids) - marker_length)
        context_length = len(prompt_ids) + len(history_ids) + len(instruction_ids)
        response_length = len(history_ids) + len(instruction_ids)
        summary_budget = min(
            int(self.config.trainer.get("gym_compaction_max_tokens", 2048)),
            self.response_length - response_length,
            self.max_model_len - context_length,
        )
        if episode_tokens_remaining is not None:
            summary_budget = min(summary_budget, episode_tokens_remaining - instruction_cost)
        if summary_budget <= 0:
            reason = (
                "episode_token_budget"
                if episode_tokens_remaining is not None and instruction_cost >= episode_tokens_remaining
                else "compaction_capacity"
            )
            return None, 0, reason

        # 4. Close the action segment and append the masked instruction to the summary.
        del action_segment.response_ids[-marker_length:]
        del action_segment.response_mask[-marker_length:]
        if action_segment.response_logprobs is not None:
            del action_segment.response_logprobs[-marker_length:]
        if action_segment.teacher_logprobs is not None:
            del action_segment.teacher_logprobs[-marker_length:]
            del action_segment.teacher_mask[-marker_length:]
        summary_segment.response_ids.extend(instruction_ids)
        summary_segment.response_mask.extend([0] * len(instruction_ids))
        summary_segment.response_logprobs.extend([0.0] * len(instruction_ids))

        # 5. Generate the summary and record only its new tokens as trainable.
        with simple_timer("generate_sequences", summary_segment.metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=summary_segment.prompt_ids + summary_segment.response_ids,
                sampling_params={**sampling_params, "max_tokens": summary_budget},
            )
        summary_ids = output.token_ids[:summary_budget]
        summary_segment.response_ids.extend(summary_ids)
        summary_segment.response_mask.extend([1] * len(summary_ids))
        if output.log_probs:
            summary_segment.response_logprobs.extend(output.log_probs[:summary_budget])
        else:
            summary_segment.response_logprobs = None
        summary_segment.routed_experts = output.routed_experts
        summary_segment.record_generation(output)
        summary_segment.summary_text = self.tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
        if log_trajectory:
            # Copied history is context here, not a second occurrence of the actions.
            summary_segment.messages = [
                *summary_messages,
                {"role": "context", "content": self.tokenizer.decode(history_ids, skip_special_tokens=False)},
                *instruction,
                {"role": "assistant", "content": self.tokenizer.decode(summary_ids, skip_special_tokens=True)},
            ]
        return summary_segment, instruction_cost + len(summary_ids), None

    async def _parse_action(self, token_ids, tool_schemas, env_name, turn):
        """Produce the environment action and the structured assistant message."""
        raw_response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        content, calls = raw_response, []
        if tool_schemas:
            content, calls = await self.tool_parser.extract_tool_calls(token_ids, tool_schemas)
        tool_calls = []
        for index, call in enumerate(calls):
            try:
                arguments = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                {
                    "name": call.name,
                    "arguments": arguments,
                    "tool_call_id": call.tool_call_id or f"call_{turn}_{index}",
                }
            )
        action = tool_calls or raw_response
        if env_name.startswith("swe_agent") and tool_schemas:
            # SWE checks the complete generated XML, including malformed/partial calls.
            action = {"raw_response": raw_response, "assistant_content": content, "tool_calls": tool_calls}
        message = {"role": "assistant", "content": content, "reasoning_content": ""}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["tool_call_id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in tool_calls
            ]
        return action, message

    async def _encode_observation(self, segment, messages, observations, tool_schemas):
        """Encode an observation and the next assistant prefix without rewriting history."""
        token_ids = segment.prompt_ids + segment.response_ids
        merged, _, _ = await self.ct_merge_non_assistant_msg(
            messages,
            messages + observations,
            token_ids,
            segment.response_mask,
            segment.response_logprobs,
            tools=tool_schemas or None,
        )
        if merged.removed_prefix_token_count:
            raise ValueError("An observation cannot remove already-recorded segment tokens.")
        return merged.token_ids[len(token_ids) :]
