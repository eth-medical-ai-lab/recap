from __future__ import annotations

import importlib
import os
from typing import Any

DEFAULT_WORKER_CLASS_PATH = "agent_system.worker.AgentEnvWorker"


def load_object(path: str) -> Any:
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


class AgentEnvWorker:
    """Run a synchronous environment behind RemoteAgentEnv's async interface."""

    def __init__(self, env_class_path: str) -> None:
        self.env_class_path = env_class_path
        self.env: Any | None = None

    def ping(self) -> bool:
        return True

    def worker_pid(self) -> int:
        return os.getpid()

    def reset(
        self,
        task_id: str | int | None,
        split: str,
        env_kwargs: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.close()
        env_class = load_object(self.env_class_path)
        self.env = env_class(split=split, **dict(env_kwargs))
        return self.env.reset(task_id=task_id, split=split)

    def step(self, action: Any) -> tuple[list[dict[str, Any]], float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        return self.env.step(action)

    def truncate(self, reason: str) -> tuple[list[dict[str, Any]], float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Call reset() before truncate().")
        return self.env.truncate(reason)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None

    def shutdown(self) -> None:
        self.close()


_WORKER_ACTOR_CLASSES: dict[str, Any] = {}


def get_worker_actor_class(worker_class_path: str = DEFAULT_WORKER_CLASS_PATH) -> Any:
    if worker_class_path not in _WORKER_ACTOR_CLASSES:
        import ray

        worker_class = load_object(worker_class_path)
        _WORKER_ACTOR_CLASSES[worker_class_path] = ray.remote(max_concurrency=1)(worker_class)
    return _WORKER_ACTOR_CLASSES[worker_class_path]
