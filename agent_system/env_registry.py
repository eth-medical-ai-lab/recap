from __future__ import annotations

from typing import Any

from agent_system.env_base import AgentEnv
from agent_system.envs.browsecomp_plus import BrowseCompPlusRemoteEnv
from agent_system.envs.swe_agent.env import SweAgentEnv

# Both supported environments are registered here, without import side effects.
_ENV_REGISTRY: dict[str, type[AgentEnv]] = {
    "browsecomp_plus_ray": BrowseCompPlusRemoteEnv,
    "swe_agent": SweAgentEnv,
}


def make_env(name: str, **kwargs: Any) -> AgentEnv:
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown agent environment {name!r}. Available: {sorted(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name](**kwargs)


def get_env_class(name: str) -> type[AgentEnv]:
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown agent environment {name!r}. Available: {sorted(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name]


def list_envs() -> list[str]:
    return sorted(_ENV_REGISTRY)


def list_task_specs(name: str, split: str = "train", **kwargs: Any) -> list[dict[str, Any]]:
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown agent environment {name!r}. Available: {sorted(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name].list_task_specs(split=split, **kwargs)

