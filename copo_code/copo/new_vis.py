import argparse
import os

from copo.algo_svo.svo_env import get_svo_env
from copo.ccenv import get_ccenv
from copo.eval.get_policy_function_from_checkpoint import get_policy_function_from_checkpoint
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv, MultiAgentRoundaboutEnv, MultiAgentTollgateEnv, \
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv, MultiAgentMetaDrive


def get_env(env, use_native_render, should_wrap_copo_env, should_wrap_cc_env):
    config = {"use_render": use_native_render}
    if "round" in env:
        env_cls = MultiAgentRoundaboutEnv
        env_name = "Round"
    elif "inter" in env:
        env_cls = MultiAgentIntersectionEnv
        env_name = "Inter"
    elif "parking" in env:
        env_cls = MultiAgentParkingLotEnv
        env_name = "Parking"
    elif "bottle" in env:
        env_cls = MultiAgentBottleneckEnv
        env_name = "Bottle"
    elif "tollgate" in env:
        env_cls = MultiAgentTollgateEnv
        env_name = "Tollgate"
    elif "pgmap" in env:
        env_cls = MultiAgentMetaDrive
        env_name = "PGMap"
    else:
        raise ValueError()

    if should_wrap_copo_env:
        assert should_wrap_cc_env is False
        env_cls = get_svo_env(get_ccenv(env_cls), return_env_class=True)
        env = env_cls(config)
        # env.set_svo_dist(mean=svo_mean, std=svo_std)

    elif should_wrap_cc_env:
        assert should_wrap_copo_env is False
        env_cls = get_ccenv(env_cls)
        env = env_cls(config)

    else:
        env = env_cls(config)

    return env, env_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="inter", type=str, choices=["inter", "round", "tollgate", "parking", "pgmap", "bottle"]
    )
    parser.add_argument("--algo", default="copo", type=str, choices=["cl", "ippo", "copo", "ccppomf", "ccppoconcat"])
    parser.add_argument("--use_native_render", action="store_true")
    args = parser.parse_args()

    # ===== Load trained policy =====
    algo = args.algo
    env = args.env

    model_name_prefix = "{}_{}".format(env, algo)

    assert os.path.isdir("new_best_checkpoints"), "Please unzip new_best_checkpoints.zip to `copo_code/copo/` folder!"
    ckpt_folder_path = None
    for p in os.listdir(os.path.abspath("new_best_checkpoints")):
        if p.startswith(model_name_prefix) and not p.endswith("metadata"):
            ckpt_folder_path = os.path.abspath(os.path.join("new_best_checkpoints", p))
            break
    assert ckpt_folder_path, f"Can't find {model_name_prefix} in {'new_best_checkpoints'}"
    succ = p.split("_")[-1]
    print(f"Found checkpoint with prefix {model_name_prefix}. The success rate should be around: {succ}")
    ckpt_path = [p for p in os.listdir(ckpt_folder_path)
                 if p.startswith("checkpoint") and not p.endswith("metadata")][0]
    ckpt_path = os.path.join(ckpt_folder_path, ckpt_path)

    should_wrap_cc_env = "ccppo" in algo
    should_wrap_copo_env = "copo" in algo

    policy_function = get_policy_function_from_checkpoint(algo=algo, ckpt=ckpt_path)

    # Note: We don't need to load LCF (local coordination factor) here since
    # we are in test-time!

    # ===== Create environment =====
    env, formal_env_name = get_env(
        env=env,
        use_native_render=args.use_native_render,
        should_wrap_copo_env=should_wrap_copo_env,
        should_wrap_cc_env=should_wrap_cc_env,
    )

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
            print("Episode success rate: ", ep_success / ep_agent if ep_agent > 0 else None)
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
                    "ep step": ep_step
                }
            )
        else:
            env.render(mode="top_down", num_stack=25)
    env.close()
