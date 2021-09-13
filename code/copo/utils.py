import gym
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env


def get_rllib_compatible_env(env_class):
    env_name = env_class.__name__

    class MA(env_class, MultiAgentEnv):
        pass

    MA.__name__ = env_name
    MA.__qualname__ = env_name
    register_env(env_name, lambda config: MA(config))
    return env_name


def validate_config_add_multiagent(config, policy_class, other_function, policy_config={}):
    from ray.tune.registry import _global_registry, ENV_CREATOR

    env_class = _global_registry.get(ENV_CREATOR, config["env"])
    single_env = env_class(config["env_config"])
    obs_space = single_env.observation_space["agent0"]
    act_space = single_env.action_space["agent0"]
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Box)
    config["multiagent"].update(
        dict(
            policies={"default": (policy_class, obs_space, act_space, policy_config)},
            policy_mapping_fn=lambda x: "default"
        )
    )
    other_function(config)
