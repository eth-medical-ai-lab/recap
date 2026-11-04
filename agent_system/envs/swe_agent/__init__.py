"""Shared SWE sandbox types; execution lives in env.py and the E2B backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SweAgentServiceError(RuntimeError):
    """A stable infrastructure or protocol failure from the sandbox service."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code

    @property
    def is_fatal(self) -> bool:
        """Shared credentials or configuration failures require stopping training."""
        return self.status_code in {401, 403} or self.kind in {
            "e2b_authentication_failed",
            "e2b_invalid_request",
            "evaluator_configuration_missing",
        }

    @property
    def is_infrastructure_failure(self) -> bool:
        """An oversized solution is an agent outcome, not a service failure."""
        return self.kind != "patch_too_large"

    def __reduce__(self):
        # BaseException only saves args (the message); Ray also needs the kind
        # and keyword-only fields to reconstruct this exception in the caller.
        return _restore_service_error, (self.kind, self.args[0], self.retryable, self.status_code)


def _restore_service_error(kind, message, retryable, status_code):
    return SweAgentServiceError(kind, message, retryable=retryable, status_code=status_code)


@dataclass(frozen=True)
class EvaluationResult:
    """Canonical task-evaluation result returned by the sandbox backend."""

    reward: float
    resolved: bool
    passed_tests: int
    total_tests: int
    output: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool = False
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    summary: str = ""


class AsyncSweAgentEpisode(Protocol):
    """One persistent repository sandbox used by a single rollout."""

    episode_id: str
    create_duration_seconds: float | None

    async def execute(self, command: str, timeout_seconds: float) -> dict[str, Any]: ...

    async def edit(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def capture_patch(self) -> dict[str, Any]: ...

    async def evaluate(
        self,
        task_metadata: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...
