"""Native Verl environment for SWE-smith training and SWE-bench evaluation."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from agent_system.env_base import AgentEnv, ChatMessage, TaskId
from agent_system.envs.swe_agent import AsyncSweAgentEpisode, SweAgentServiceError
from agent_system.envs.swe_agent.prompts import (
    BASH_TOOL_NAME,
    EDITOR_TOOL_NAME,
    FINISH_TOOL_NAME,
    QWEN35_MODEL_REVISION,
    ToolAction,
    ToolCallError,
    forbidden_action,
    invalid_tool_message,
    parse_tool_action,
    render_initial_messages,
    render_observation,
    tool_schemas,
)
from agent_system.envs.swe_agent.tasks import (
    SweBenchVerifiedTaskStore,
    load_swe_smith_task,
    load_swe_smith_task_at,
    make_swebench_solution,
    resolve_swe_smith_data_file,
    swe_smith_task_offsets,
)

PROTOCOL_NAME = "swe_agent_qwen35_native_tools"
SWE_SMITH_DATASET = "swe_smith"
SWE_BENCH_VERIFIED = "swebench_verified"
logger = logging.getLogger(__name__)


class SweAgentEnv(AgentEnv):
    """Run one native-tool SWE episode in an E2B sandbox."""

    # Git is enabled only when the backend provides an agent-visible checkout.
    allow_git = False
    filter_network_text = True

    def __init__(
        self,
        data_path: str,
        dataset: str = SWE_SMITH_DATASET,
        split: str = "train",
        max_turns: int = 40,
        command_timeout_seconds: float = 120,
        evaluation_timeout_seconds: float = 600,
        observation_char_cap: int = 10_000,
        max_format_errors: int = 3,
        e2b_timeout_seconds: float = 3600,
        e2b_run_id: str | None = None,
        e2b_create_concurrency: int = 4,
        e2b_create_max_retries: int = 4,
        e2b_create_retry_base_seconds: float = 1.0,
        native_tool_calls: bool = False,
        model_name: str = f"Qwen/Qwen3.5-4B@{QWEN35_MODEL_REVISION}",
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if command_timeout_seconds <= 0 or evaluation_timeout_seconds <= 0:
            raise ValueError("command and evaluation timeouts must be positive")
        if observation_char_cap <= 0:
            raise ValueError("observation_char_cap must be positive")
        if max_format_errors < 0:
            raise ValueError("max_format_errors must be non-negative")

        if dataset not in {SWE_SMITH_DATASET, SWE_BENCH_VERIFIED}:
            raise ValueError(f"Unknown SWE dataset {dataset!r}.")
        if dataset == SWE_BENCH_VERIFIED and split != "test":
            raise ValueError("SWE-bench Verified is evaluation-only and requires split='test'.")

        self.data_path = data_path
        self.native_tool_calls = native_tool_calls
        self.model_name = model_name
        self.protocol_name = "swe_agent_native_tools" if native_tool_calls else PROTOCOL_NAME
        self.dataset = dataset
        self.allow_git = self.allow_git or (dataset == SWE_BENCH_VERIFIED)
        self.split = split
        self.max_turns = max_turns
        self.command_timeout_seconds = command_timeout_seconds
        self.evaluation_timeout_seconds = evaluation_timeout_seconds
        self.observation_char_cap = observation_char_cap
        self.max_format_errors = max_format_errors
        # E2BEpisode.create validates these settings before allocating a sandbox.
        self.e2b_timeout_seconds = e2b_timeout_seconds
        self.e2b_run_id = e2b_run_id
        self.e2b_create_concurrency = e2b_create_concurrency
        self.e2b_create_max_retries = e2b_create_max_retries
        self.e2b_create_retry_base_seconds = e2b_create_retry_base_seconds
        self._verified_store = (
            SweBenchVerifiedTaskStore(data_path) if dataset == SWE_BENCH_VERIFIED else None
        )

        self.task: dict[str, Any] | None = None
        self.private_task: dict[str, Any] | None = None
        self.task_id: str | None = None
        self.episode_id: str | None = None
        self.episode: AsyncSweAgentEpisode | None = None
        self.turn_count = 0
        self.consecutive_format_errors = 0
        self._finished = False

    @classmethod
    def list_task_specs(cls, split: str = "train", **kwargs: Any) -> list[dict[str, Any]]:
        data_path = kwargs.get("data_path")
        if not data_path:
            raise ValueError("SweAgentEnv.list_task_specs requires data_path.")
        dataset = str(kwargs.get("dataset", SWE_SMITH_DATASET))
        if dataset == SWE_BENCH_VERIFIED:
            if split != "test":
                raise ValueError("SWE-bench Verified is evaluation-only and requires split='test'.")
            return SweBenchVerifiedTaskStore(data_path).list_task_specs()
        if dataset != SWE_SMITH_DATASET:
            raise ValueError(f"Unknown SWE dataset {dataset!r}.")

        data_file = resolve_swe_smith_data_file(str(data_path), split)
        specs = []
        for instance_id, location in swe_smith_task_offsets(data_file).items():
            task = load_swe_smith_task_at(data_file, *location)
            specs.append(
                {
                    "env_name": "swe_agent",
                    "split": split,
                    "task_id": instance_id,
                    "uid": f"swe_agent:{split}:{instance_id}",
                    "image_name": str(task["image_name"]),
                    "template_name": str(task["template_name"]),
                    "repo": str(task["repo"]),
                }
            )
        return specs

    async def reset(
        self,
        task_id: TaskId | None = None,
        split: str | None = None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        """Start a fresh sandbox and return the task prompt with fixed episode metadata."""
        # 1. Close the previous episode and load the task and private grading data.
        await self.close()
        if split is not None:
            if self.dataset == SWE_BENCH_VERIFIED and split != "test":
                raise ValueError("SWE-bench Verified is evaluation-only and requires split='test'.")
            self.split = split

        if self._verified_store is not None:
            requested_id = str(task_id) if task_id is not None else self._verified_store.task_ids[0]
            task = self._verified_store.load_rollout_task(requested_id)
            private_task = self._verified_store.load_private_task(requested_id)
            checkout_ref = str(task["base_commit"])
        else:
            data_file = resolve_swe_smith_data_file(self.data_path, self.split)
            offsets = swe_smith_task_offsets(data_file)
            requested_id = str(task_id) if task_id is not None else next(iter(offsets))
            task = load_swe_smith_task(data_file, requested_id)
            private_task = task
            checkout_ref = None

        # 2. Create the sandbox and initialize episode state after creation succeeds.
        from agent_system.envs.swe_agent.e2b_episode import E2BEpisode

        episode = await E2BEpisode.create(
            template_name=str(task["template_name"]),
            task_id=requested_id,
            image_name=str(task["image_name"]),
            checkout_ref=checkout_ref or requested_id,
            allow_git=self.allow_git,
            timeout_seconds=self.e2b_timeout_seconds,
            run_id=self.e2b_run_id,
            create_concurrency=self.e2b_create_concurrency,
            create_max_retries=self.e2b_create_max_retries,
            create_retry_base_seconds=self.e2b_create_retry_base_seconds,
        )

        episode_id = episode.episode_id

        self.task = task
        self.private_task = private_task
        self.task_id = requested_id
        self.episode_id = episode_id
        self.episode = episode
        self.turn_count = 0
        self.consecutive_format_errors = 0
        self._finished = False

        # 3. Return the initial messages and fixed metadata; grading data stays private.
        reset_info = {
            "env_name": "swe_agent",
            "protocol": self.protocol_name,
            "allow_git": self.allow_git,
            "tool_schemas": tool_schemas(),
            "dataset": self.dataset,
            "split": self.split,
            "task_id": requested_id,
            "instance_id": requested_id,
            "repo": str(task["repo"]),
            "image_name": str(task["image_name"]),
            "template_name": str(task["template_name"]),
            "episode_id": episode_id,
            "max_turns": self.max_turns,
            "command_timeout_seconds": self.command_timeout_seconds,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "observation_char_cap": self.observation_char_cap,
            "max_format_errors": self.max_format_errors,
            "e2b_timeout_seconds": self.e2b_timeout_seconds,
            "sandbox_create_duration_seconds": episode.create_duration_seconds,
            "e2b_run_id": self.e2b_run_id,
            "e2b_create_concurrency": self.e2b_create_concurrency,
            "e2b_create_max_retries": self.e2b_create_max_retries,
            "e2b_create_retry_base_seconds": self.e2b_create_retry_base_seconds,
            "e2b_create_attempts": getattr(episode, "create_attempts", 1),
            "e2b_create_retries": getattr(episode, "create_retries", 0),
            "e2b_create_limiter_wait_seconds": getattr(episode, "create_limiter_wait_seconds", 0.0),
            "e2b_create_rate_limit_wait_seconds": getattr(episode, "create_rate_limit_wait_seconds", 0.0),
        }
        if self._verified_store is None:
            reset_info.update(
                {
                    "fail_to_pass_count": len(task["FAIL_TO_PASS"]),
                    "pass_to_pass_count": len(task["PASS_TO_PASS"]),
                }
            )
        return render_initial_messages(str(task["problem_statement"]), allow_git=self.allow_git), reset_info

    async def step(self, action: Any) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        """Execute one action and check termination; invalid actions also consume a turn."""
        # 1. Validate episode state and parse one action.
        if self.task is None or self.task_id is None or self.episode_id is None or self.episode is None:
            raise RuntimeError("Call reset() before step().")
        if self._finished:
            raise RuntimeError("The SWE-agent episode is already finished.")

        self.turn_count += 1
        try:
            tool_action = parse_tool_action(action, require_qwen_xml=not self.native_tool_calls)
        except ToolCallError as error:
            # Invalid actions count toward both limits but never execute a tool.
            self.consecutive_format_errors += 1
            info = self._step_info(format_ok=False)
            info.update(
                {
                    "action_type": "invalid_tool_call",
                    "raw_action": action,
                    "tool_calls": [],
                    "error": str(error),
                }
            )
            format_limit = self.max_format_errors > 0 and self.consecutive_format_errors >= self.max_format_errors
            observation = [] if format_limit else [{"role": "user", "content": invalid_tool_message(str(error))}]
            if format_limit or self.turn_count >= self.max_turns:
                reason = "format_error_limit" if format_limit else "max_turns"
                return await self._finish(observation, info, submitted=False, reason=reason)
            return observation, 0.0, False, info

        self.consecutive_format_errors = 0
        info = self._step_info(format_ok=True)
        info.update(
            {
                "action_type": tool_action.name,
                "raw_action": action,
                "tool_calls": [tool_action.name],
                "tool_call_id": tool_action.tool_call_id,
                "has_rationale": tool_action.has_rationale,
            }
        )

        # 2. Execute the requested tool, or grade an explicit finish immediately.
        if tool_action.name == FINISH_TOOL_NAME:
            observation = [_tool_message(tool_action, "Task submitted for evaluation.")]
            return await self._finish(observation, info, submitted=True, reason="finish")

        if tool_action.name == BASH_TOOL_NAME:
            command = str(tool_action.arguments["command"])
            blocked_reason = forbidden_action(
                command, allow_git=self.allow_git, filter_network_text=self.filter_network_text
            )
            if blocked_reason is not None:
                observation_text = render_observation(
                    1,
                    blocked_reason,
                    self.observation_char_cap,
                )
                observation = [_tool_message(tool_action, observation_text)]
                info.update(
                    {
                        "action_type": "forbidden",
                        "command": command,
                        "return_code": 1,
                        "observation_text": observation_text,
                    }
                )
            else:
                result = await self.episode.execute(
                    command,
                    self.command_timeout_seconds,
                )
                command_result = _validate_command_result(result)
                output = (
                    f"[timed out after {self.command_timeout_seconds:g}s]"
                    if command_result["timed_out"]
                    else command_result["stdout"] + command_result["stderr"]
                )
                observation_text = render_observation(
                    command_result["return_code"],
                    output,
                    self.observation_char_cap,
                )
                observation = [_tool_message(tool_action, observation_text)]
                info.update(
                    {
                        "command": command,
                        "command_result": command_result,
                        "return_code": command_result["return_code"],
                        "timed_out": command_result["timed_out"],
                        "observation_text": observation_text,
                        "observation_chars": len(output),
                        "observation_truncated": (
                            command_result["output_truncated"] or len(output) > self.observation_char_cap
                        ),
                        "observation_elided_chars": max(0, len(output) - self.observation_char_cap),
                    }
                )
        else:
            assert tool_action.name == EDITOR_TOOL_NAME
            editor_result = await self.episode.edit(tool_action.arguments)
            output = editor_result.get("output")
            if not isinstance(output, str) or not isinstance(editor_result.get("error"), bool):
                raise _protocol_error("edit result has invalid fields")
            observation = [_tool_message(tool_action, output)]
            editor_duration = editor_result.get("duration_seconds")
            info.update(
                {
                    "editor_command": tool_action.arguments["command"],
                    "editor_path": tool_action.arguments["path"],
                    "editor_error": editor_result["error"],
                    "observation_text": output,
                    "sandbox_editor_duration_seconds": editor_duration,
                }
            )

        # 3. Apply the turn limit after execution so the final action affects grading.
        if self.turn_count >= self.max_turns:
            return await self._finish(observation, info, submitted=False, reason="max_turns")
        return observation, 0.0, False, info

    async def truncate(
        self,
        reason: str,
    ) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        """Evaluate the current patch when an outer rollout limit is reached."""
        if self.task is None or self.task_id is None or self.episode_id is None or self.episode is None:
            raise RuntimeError("Call reset() before truncate().")
        if self._finished:
            raise RuntimeError("The SWE-agent episode is already finished.")
        if not isinstance(reason, str) or not reason:
            raise ValueError("truncate reason must be a nonempty string")

        info = self._step_info(format_ok=None)
        info.update(
            {
                "action_type": "rollout_limit",
                "raw_action": None,
                "tool_calls": [],
            }
        )
        return await self._finish([], info, submitted=False, reason=reason)

    async def close(self) -> None:
        episode = self.episode
        if episode is not None:
            await episode.close()
        self.task = None
        self.private_task = None
        self.task_id = None
        self.episode_id = None
        self.episode = None
        self.turn_count = 0
        self.consecutive_format_errors = 0
        self._finished = False

    async def _finish(
        self,
        observation: list[ChatMessage],
        info: dict[str, Any],
        *,
        submitted: bool,
        reason: str,
    ) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        """Capture and grade the current patch, then return the terminal task result."""
        assert self.task is not None and self.private_task is not None
        assert self.episode_id is not None
        assert self.episode is not None

        # 1. Capture the current solution once for all grading attempts.
        patch_result = await self.episode.capture_patch()
        patch = patch_result.get("patch")
        if not isinstance(patch, str):
            raise _protocol_error("capture_patch result is missing patch")

        # 2. Grade the saved patch in a fresh sandbox, with at most three attempts.
        for attempt in range(3):
            try:
                evaluation = await self.episode.evaluate(
                    self._evaluation_metadata(),
                    self.evaluation_timeout_seconds,
                )
                break
            except SweAgentServiceError as error:
                if not error.retryable or error.is_fatal:
                    raise
                if attempt == 2:
                    # The grading budget is exhausted; do not also restart the
                    # entire trajectory in GymAgentLoop.
                    raise SweAgentServiceError(
                        error.kind,
                        str(error),
                        retryable=False,
                        status_code=error.status_code,
                    ) from error
                logger.warning("Retrying SWE evaluation task=%s attempt=%s: %s", self.task_id, attempt + 2, error)
                await asyncio.sleep(2**attempt)
        evaluation = _validate_evaluation_result(evaluation)

        # 3. Build the terminal result and optional SWE-bench solution record.
        reward = evaluation["reward"]
        truncated = not submitted
        self._finished = True

        info.update(
            {
                "completed": True,
                "status": reason,
                "termination_reason": reason,
                "submitted": submitted,
                "terminated": submitted,
                "truncated": truncated,
                "reward": reward,
                "reward_pending": False,
                "patch": patch,
                "patch_bytes": len(patch.encode("utf-8")),
                "patch_capture_duration_seconds": patch_result.get("duration_seconds"),
                "evaluation": evaluation,
                "evaluation_stats": {
                    "success": evaluation["resolved"],
                    "passed_tests": evaluation["passed_tests"],
                    "total_tests": evaluation["total_tests"],
                    "timed_out": evaluation["timed_out"],
                    "duration_seconds": evaluation["duration_seconds"],
                    "summary": evaluation.get("summary", ""),
                },
            }
        )
        if self._verified_store is not None:
            info["swebench_solution"] = make_swebench_solution(
                str(self.task["instance_id"]),
                patch,
                self.model_name,
            )
        return observation, reward, True, info

    def _evaluation_metadata(self) -> dict[str, Any]:
        assert self.private_task is not None
        if self._verified_store is not None:
            metadata = dict(self.private_task)
            metadata["dataset_type"] = SWE_BENCH_VERIFIED
            return metadata
        return {
            "dataset_type": SWE_SMITH_DATASET,
            "instance_id": str(self.private_task["instance_id"]),
            "repo": str(self.private_task["repo"]),
            "FAIL_TO_PASS": list(self.private_task["FAIL_TO_PASS"]),
            "PASS_TO_PASS": list(self.private_task["PASS_TO_PASS"]),
        }

    def _step_info(self, *, format_ok: bool | None) -> dict[str, Any]:
        """Return changing counters; fixed task/configuration metadata belongs to reset()."""
        info = {
            "turn_count": self.turn_count,
            "consecutive_format_errors": self.consecutive_format_errors,
        }
        if format_ok is not None:
            info["format_ok"] = format_ok
        return info


def _validate_command_result(result: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("stdout"), str)
        or not isinstance(result.get("stderr"), str)
        or isinstance(result.get("return_code"), bool)
        or not isinstance(result.get("return_code"), int)
        or not isinstance(result.get("timed_out"), bool)
        or not _is_finite_number(result.get("duration_seconds"), minimum=0)
        or ("output_truncated" in result and not isinstance(result["output_truncated"], bool))
    ):
        raise _protocol_error("exec result has invalid fields")
    validated = dict(result)
    validated["output_truncated"] = bool(result.get("output_truncated", False))
    return validated


def _validate_evaluation_result(result: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(result, dict)
        or not _is_finite_number(result.get("reward"))
        or not isinstance(result.get("resolved"), bool)
        or not _is_nonnegative_int(result.get("passed_tests"))
        or not _is_nonnegative_int(result.get("total_tests"))
        or result.get("passed_tests", 0) > result.get("total_tests", 0)
        or not isinstance(result.get("output"), str)
        or not isinstance(result.get("timed_out"), bool)
        or not _is_finite_number(result.get("duration_seconds"), minimum=0)
        or ("summary" in result and not isinstance(result["summary"], str))
        or ("output_truncated" in result and not isinstance(result["output_truncated"], bool))
    ):
        raise _protocol_error("evaluate result has invalid fields")
    validated = dict(result)
    validated["reward"] = float(result["reward"])
    return validated


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and (minimum is None or number >= minimum)


def _tool_message(action: ToolAction, content: str) -> ChatMessage:
    return {
        "role": "tool",
        "name": action.name,
        "tool_call_id": action.tool_call_id,
        "content": content,
    }


def _protocol_error(message: str) -> SweAgentServiceError:
    return SweAgentServiceError("service_protocol_error", message, retryable=False)
