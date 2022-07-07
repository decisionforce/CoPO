"""
Tutorial:
    1. Use CCPPOCurriculum
    2. Use ChangeNCallback
"""

import copy

from copo.algo_ippo.ippo import IPPOTrainer, validate_config_add_multiagent, PPO_valid, PPOTFPolicy
from copo.callbacks import MultiAgentDrivingCallbacks


def validate_config(config):
    # config["target_num_agents"] = config["env_config"]["num_agents"]
    # config["env_config"]["num_agents"] = 1
    validate_config_add_multiagent(config, PPOTFPolicy, PPO_valid)


IPPOCL = IPPOTrainer.with_updates(name="IPPOCL", validate_config=validate_config)


def get_change_n_env(env_class):
    class ChangeNEnv(env_class):
        def __init__(self, config):
            self._raw_input_config = copy.deepcopy(config)
            super(ChangeNEnv, self).__init__(config)

        def close_and_reset_num_agents(self, num_agents):
            config = copy.deepcopy(self._raw_input_config)
            self.close()
            config["num_agents"] = num_agents
            super(ChangeNEnv, self).__init__(config)

    name = env_class.__name__
    name = "CL{}".format(name)
    ChangeNEnv.__name__ = name
    ChangeNEnv.__qualname__ = name
    return ChangeNEnv


def get_change_n_callback(total_time_step):
    class ChangeNCallback(MultiAgentDrivingCallbacks):
        def __init__(self):
            super(ChangeNCallback, self).__init__()
            self.target_num_agents = None
            self.should_set = False
            self.last_steps = 0
            self.total_time_step = total_time_step  # Set total timesteps here!

        def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
            if self.target_num_agents is None:
                self.target_num_agents = base_env.envs[0].default_config()["num_agents"]
            super(ChangeNCallback, self).on_episode_end(worker, base_env, policies, episode, **kwargs)
            if worker.global_vars is None:
                return
            current_steps = worker.global_vars["timestep"]
            if self.last_steps <= self.total_time_step / 4 * 1 < current_steps:
                num_agents = int(self.target_num_agents / 4 * 2)
                self.set_envs_num_agents(num_agents, base_env, current_steps)
            elif self.last_steps <= self.total_time_step / 4 * 2 < current_steps:
                num_agents = int(self.target_num_agents / 4 * 3)
                self.set_envs_num_agents(num_agents, base_env, current_steps)
            elif self.last_steps <= self.total_time_step / 4 * 3 < current_steps:
                num_agents = int(self.target_num_agents / 4 * 4)
                self.set_envs_num_agents(num_agents, base_env, current_steps)
            if current_steps <= self.total_time_step / 4 * 1 and self.last_steps == 0:
                num_agents = int(self.target_num_agents / 4 * 1)
                self.set_envs_num_agents(num_agents, base_env, current_steps)
            self.last_steps = current_steps

        def set_envs_num_agents(self, num_agents, base_env, t):
            print('Current time step: {}. We are now setting all environments with {} agents!'.format(t, num_agents))
            for e in base_env.envs:
                e.close_and_reset_num_agents(num_agents)
                e.reset()
                print("Hi!!! We are in environment now! Current agents: ", len(e.vehicles.keys()))

    return ChangeNCallback


if __name__ == '__main__':
    # Testing only!
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
    from copo.train.train import train
    from copo.train.utils import get_train_parser
    from copo.utils import get_rllib_compatible_env
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv

    parser = get_train_parser()
    args = parser.parse_args()

    total_timesteps = 10000

    stop = {"timesteps_total": total_timesteps}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    config = dict(
        env=get_rllib_compatible_env(get_change_n_env(MultiAgentRoundaboutEnv)),
        env_config=dict(
            max_step_per_agent=100,
            # start_seed=tune.grid_search([5000]),
            # num_agents=12,
            # crash_done=True,
        ),
        num_sgd_iter=1,
        rollout_fragment_length=20,
        train_batch_size=40,
        sgd_minibatch_size=20,
        num_workers=0,
        # counterfactual=True,
        # fuse_mode="mf",
    )
    results = train(
        IPPOCL,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=0,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=args.num_gpus,
        num_seeds=1,
        max_failures=0,
        exp_name=exp_name,
        custom_callback=get_change_n_callback(total_timesteps),
        test_mode=True,
        local_mode=True
    )
