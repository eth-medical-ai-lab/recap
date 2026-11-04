from __future__ import annotations

import asyncio
from typing import Any

from agent_system.env_base import AgentEnv, ChatMessage, TaskId
from agent_system.worker import DEFAULT_WORKER_CLASS_PATH, get_worker_actor_class, load_object


class AgentWorkerPool:
    def __init__(
        self,
        pool_size: int,
        worker_class_path: str = DEFAULT_WORKER_CLASS_PATH,
        worker_num_cpus: float = 0.1,
        worker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError("Agent worker pool size must be at least 1.")
        if worker_num_cpus <= 0:
            raise ValueError("Agent worker num_cpus must be positive.")

        self.pool_size = pool_size
        self.worker_class_path = worker_class_path
        self.worker_num_cpus = worker_num_cpus
        self.worker_kwargs = dict(worker_kwargs or {})
        self._ready: list[Any] = []
        self._leased: list[Any] = []
        self._started = False
        self._condition = asyncio.Condition()

    async def acquire(self) -> Any:
        async with self._condition:
            await self._start()
            while not self._ready:
                await self._condition.wait()
            worker = self._ready.pop()
            self._leased.append(worker)
            return worker

    async def release(self, worker: Any) -> None:
        async with self._condition:
            if worker not in self._leased:
                return

            self._leased.remove(worker)
            self._ready.append(worker)
            self._condition.notify()

    async def replace(self, worker: Any) -> None:
        import ray

        async with self._condition:
            if worker in self._leased:
                self._leased.remove(worker)
            if worker in self._ready:
                self._ready.remove(worker)
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
            self._ready.append(self._make_worker())
            self._condition.notify()

    async def _start(self) -> None:
        if self._started:
            return

        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        for _ in range(self.pool_size):
            self._ready.append(self._make_worker())
        self._started = True

    def _make_worker(self) -> Any:
        worker_class = get_worker_actor_class(self.worker_class_path)
        return worker_class.options(num_cpus=self.worker_num_cpus).remote(**self.worker_kwargs)


_POOLS: dict[tuple[Any, ...], AgentWorkerPool] = {}


def get_agent_worker_pool(
    pool_size: int,
    worker_class_path: str = DEFAULT_WORKER_CLASS_PATH,
    worker_num_cpus: float = 0.1,
    worker_kwargs: dict[str, Any] | None = None,
) -> AgentWorkerPool:
    worker_kwargs = dict(worker_kwargs or {})
    key = (
        pool_size,
        worker_class_path,
        worker_num_cpus,
        _freeze(worker_kwargs),
    )
    if key not in _POOLS:
        _POOLS[key] = AgentWorkerPool(
            pool_size=pool_size,
            worker_class_path=worker_class_path,
            worker_num_cpus=worker_num_cpus,
            worker_kwargs=worker_kwargs,
        )
    return _POOLS[key]


class RemoteAgentEnv(AgentEnv):
    env_name = ""
    registered_name = ""
    env_class_path = ""
    worker_class_path = DEFAULT_WORKER_CLASS_PATH
    worker_kwargs: dict[str, Any] = {}
    default_env_kwargs: dict[str, Any] = {}
    default_ray_pool_size = 4
    default_ray_worker_num_cpus = 0.1

    def __init__(
        self,
        split: str = "train",
        ray_pool_size: int | None = None,
        ray_worker_num_cpus: float | None = None,
        **env_kwargs: Any,
    ) -> None:
        if not self.env_class_path:
            raise ValueError(f"{type(self).__name__}.env_class_path is not set.")

        self.split = split
        self.env_kwargs = dict(self.default_env_kwargs)
        self.env_kwargs.update(env_kwargs)

        worker_kwargs = self._worker_kwargs()
        self.pool = get_agent_worker_pool(
            pool_size=self.default_ray_pool_size if ray_pool_size is None else ray_pool_size,
            worker_class_path=self.worker_class_path,
            worker_num_cpus=(self.default_ray_worker_num_cpus if ray_worker_num_cpus is None else ray_worker_num_cpus),
            worker_kwargs=worker_kwargs,
        )
        self.worker: Any | None = None

    @classmethod
    def list_task_specs(cls, split: str = "train", **kwargs: Any) -> list[dict[str, Any]]:
        env_class = load_object(cls.env_class_path)
        specs = env_class.list_task_specs(split=split, **cls._env_kwargs_for_specs(kwargs))
        public_name = cls.registered_name or cls.env_name
        if not public_name:
            return specs

        updated_specs = []
        for spec in specs:
            spec = dict(spec)
            spec["env_name"] = public_name
            spec["uid"] = f"{public_name}:{split}:{spec['task_id']}"
            updated_specs.append(spec)
        return updated_specs

    async def reset(
        self,
        task_id: TaskId | None = None,
        split: str | None = None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        await self.close()
        if split is not None:
            self.split = split

        self.worker = await self.pool.acquire()
        try:
            return await self.worker.reset.remote(task_id, self.split, self.env_kwargs)
        except Exception:
            worker = self.worker
            self.worker = None
            await self.pool.replace(worker)
            raise

    async def step(self, action: Any) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        if self.worker is None:
            raise RuntimeError("Call reset() before step().")
        try:
            return await self.worker.step.remote(action)
        except Exception:
            worker = self.worker
            self.worker = None
            await self.pool.replace(worker)
            raise

    async def truncate(self, reason: str) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        if self.worker is None:
            raise RuntimeError("Call reset() before truncate().")
        try:
            return await self.worker.truncate.remote(reason)
        except Exception:
            worker = self.worker
            self.worker = None
            await self.pool.replace(worker)
            raise

    async def close(self) -> None:
        if self.worker is None:
            return

        worker = self.worker
        self.worker = None
        try:
            await worker.close.remote()
        except Exception:
            await self.pool.replace(worker)
            return
        await self.pool.release(worker)

    def _worker_kwargs(self) -> dict[str, Any]:
        worker_kwargs = dict(self.worker_kwargs)
        if self.worker_class_path == DEFAULT_WORKER_CLASS_PATH:
            worker_kwargs.setdefault("env_class_path", self.env_class_path)
        return worker_kwargs

    @classmethod
    def _env_kwargs_for_specs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        env_kwargs = dict(cls.default_env_kwargs)
        env_kwargs.update(kwargs)
        env_kwargs.pop("ray_pool_size", None)
        env_kwargs.pop("ray_worker_num_cpus", None)
        return env_kwargs


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value
