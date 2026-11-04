from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

TaskId = str | int
ChatMessage = dict[str, Any]


class AgentEnv(ABC):
    """Async episode interface used by the agent loop.

    step() and truncate() return observations, reward, done, and info. done means
    no further actions are allowed; info distinguishes termination from truncation.
    reset() returns fixed task/configuration metadata; step/truncate info contains
    changing counters, action diagnostics, and final results.
    close() releases resources without scoring the episode.
    """

    @classmethod
    def list_task_specs(cls, split: str = "train", **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def reset(
        self,
        task_id: TaskId | None = None,
        split: str | None = None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def step(self, action: Any) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def truncate(self, reason: str) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        """Finalize an unfinished episode at a rollout limit, returning done=True."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release episode resources; repeated calls are safe."""
        raise NotImplementedError
