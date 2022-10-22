import argparse

from copo.eval.get_policy_function import PolicyFunction
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="inter", type=str)
    parser.add_argument("--algo", default="copo", type=str)
    parser.add_argument("--use_native_render", action="store_true")
    args = parser.parse_args()

    # ===== Load trained policy =====
    assert args.algo in ["cl", "copo", "ippo", "ccppo"]
    model_name = "{}_{}".format(args.algo, args.env)
    policy_function = PolicyFunction(model_name=model_name)

    # ===== Create environment =====
    env = get_env(args.env, use_native_render=args.use_native_render)

    # ===== Running =====
    o = env.reset()
    d = {"__all__": False}
    ep_success = 0
    ep_step = 0
    ep_agent = 0
    for i in range(1, 100000):
        action = policy_function(o, d)
        o, r, d, info = env.step(action)
        ep_step += 1
        for kkk, ddd in d.items():
            if kkk != "__all__" and ddd:
                ep_success += 1 if info[kkk]["arrive_dest"] else 0
                ep_agent += 1
        if d["__all__"]:  # This is important!
            print(d, info)
            print(
                {
                    "total agents": ep_agent,
                    "existing agents": len(o),
                    "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                    "ep step": ep_step
                }
            )
            ep_success = 0
            ep_step = 0
            ep_agent = 0
            o = env.reset()
            d = {"__all__": False}
            policy_function.reset()
            # break
        if args.use_native_render:
            env.render(
                text={
                    "total agents": ep_agent,
                    "existing agents": len(o),
                    "success rate": ep_success / ep_agent if ep_agent > 0 else None,
                    "ep step": ep_step,
                    "Press": "Q to switch view"
                }
            )
        else:
            env.render(mode="top_down", num_stack=25)
    env.close()
