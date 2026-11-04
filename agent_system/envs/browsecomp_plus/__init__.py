from agent_system.pool import RemoteAgentEnv


class BrowseCompPlusRemoteEnv(RemoteAgentEnv):
    env_name = "browsecomp_plus"
    registered_name = "browsecomp_plus_ray"
    env_class_path = "agent_system.envs.browsecomp_plus.env.BrowseCompPlusEnv"


__all__ = ["BrowseCompPlusRemoteEnv"]
