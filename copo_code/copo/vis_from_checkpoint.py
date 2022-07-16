"""
We provide a script to demonstrate how to directly
visualize population behavior from RLLib checkpoint.
Please change the `ckpt_path` in the following to your
own checkpoint path.

Note that if you are restoring CoPO checkpoint, you need to implement appropriate
wrapper to encode the LCF into the observation and feed them to the neural network.
"""
from copo.eval.get_policy_function import PolicyFunction, _compute_actions_for_torch_policy, \
    _compute_actions_for_tf_policy
from metadrive.envs.marl_envs import MultiAgentBottleneckEnv, MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, \
    MultiAgentParkingLotEnv, MultiAgentTollgateEnv


def get_env(env_name, use_native_render=False):
    config = {"use_render": use_native_render}
    if env_name == "inter":
        return MultiAgentIntersectionEnv(config)
    elif env_name == "round":
        return MultiAgentRoundaboutEnv(config)
    elif env_name == "parking":
        return MultiAgentParkingLotEnv(config)
    elif env_name == "tollgate":
        return MultiAgentTollgateEnv(config)
    elif env_name == "bottle":
        return MultiAgentBottleneckEnv(config)
    else:
        raise ValueError("Unknown environment {}!".format(env_name))


if __name__ == "__main__":
    # ===== Specify the details =====
    ckpt_path = "path_my_my_ckpt/checkpoint_1234/checkpoint-1234"
    env_name = "inter"
    using_torch_policy = False
    policy_name = "default"
    use_native_render = True  # Set to False to use Pygame Renderer
    deterministic = False

    # ===== Load trained policy =====
    # import pickle

    # If pickle can not be imported, try: pip install pickle5
    # and use:
    import pickle5 as pickle

    if using_torch_policy:
        policy_class = _compute_actions_for_torch_policy
    else:
        policy_class = _compute_actions_for_tf_policy

    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"][policy_name]:
        worker["state"][policy_name].pop("_optimizer_variables")
    pickled_worker = pickle.dumps(worker)
    weights = worker["state"][policy_name]
    # remove value network
    if using_torch_policy:
        weights = weights["weights"]
        weights = {k: v for k, v in weights.items() if "value" not in k}
    else:
        weights = {k: v for k, v in weights.items() if "value" not in k}

    def policy(obs):
        ret = policy_class(weights, obs, policy_name=policy_name, layer_name_suffix="_1", deterministic=deterministic)
        return ret

    policy_function = PolicyFunction(policy=policy)

    # ===== Create environment =====
    env = get_env(env_name, use_native_render=use_native_render)

    # ===== Running =====
    o = env.reset()
    # env.pg_world.force_fps.toggle()  # Uncomment this line to accelerate
    d = {"__all__": False}
    ep_success = 0
    ep_step = 0
    ep_agent = 0
    ep_done = 0
    ep_reward_sum = 0.0
    ep_success_reward_sum = 0.0
    for i in range(1, 100000):
        action = policy_function(o, d)
        o, r, d, info = env.step(action)
        ep_step += 1
        for k, ddd in d.items():
            if ddd and k in info:
                ep_success += int(info[k]["arrive_dest"])
                ep_reward_sum += int(info[k]["episode_reward"])
                ep_done += 1
                if info[k]["arrive_dest"]:
                    ep_success_reward_sum += int(info[k]["episode_reward"])
        if d["__all__"]:  # This is important!
            print(d, info)
            print(
                "Success Rate: {:.3f}, reward: {:.3f}, success reward: {:.3f}, failed reward: {:.3f}, total num {}".
                format(
                    ep_success / ep_done if ep_done > 0 else -1, ep_reward_sum / ep_done if ep_done > 0 else -1,
                    ep_success_reward_sum / ep_success if ep_success > 0 else -1,
                    (ep_reward_sum - ep_success_reward_sum) / (ep_done - ep_success) if
                    (ep_done - ep_success) > 0 else -1, ep_done
                )
            )
            ep_success = 0
            ep_step = 0
            ep_agent = 0
            o = env.reset()
            d = {"__all__": False}
            policy_function.reset()
            break
        if use_native_render:
            env.render(
                text={
                    "total agents": ep_agent,
                    "existing agents": len(o),
                    "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                    "ep step": ep_step
                }
            )
        else:
            env.render(mode="top_down", num_stack=25)
    env.close()
