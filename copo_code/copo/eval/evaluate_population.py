"""
Zhenghao: This script is not intended to be used formally by other users since I just used it to evaluate my trained
populations. You can use it directly to evaluate provided checkpoints.

For your own population, please refer to copo_code/copo/eval.py for a formal script to evaluate!
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from copo import pretty_print
from copo.eval.get_policy_function import PolicyFunction
from copo.eval.recoder import RecorderEnv
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv, MultiAgentRoundaboutEnv, MultiAgentTollgateEnv, \
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv


def evaluate_once(
    model_name, make_env, num_episodes=10, use_distributional_svo=False, suffix="", auto_add_svo_to_obs=True
):
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    # Setup policy
    # try:
    policy_function = PolicyFunction(
        model_name,
        use_distributional_svo=use_distributional_svo and model_name.startswith("metasvo"),
        auto_add_svo_to_obs=auto_add_svo_to_obs
    )
    # except FileNotFoundError:
    #     print("We failed to load data with model name: ", model_name)
    #     return None

    # Setup environment
    env = make_env()
    try:
        o = env.reset()
        d = {"__all__": False}
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []
        while True:

            # Step the environment
            o, r, d, info = env.step(policy_function(o, d))
            step_count += 1

            if step_count % 100 == 0:
                print(
                    "Evaluating {}, Num episodes: {}, Num steps in this episode: {} (Ep time {:.2f}, "
                    "Total time {:.2f})".format(
                        model_name, ep_count, step_count, np.mean(ep_times),
                        time.time() - start
                    )
                )

            # Reset the environment
            if d["__all__"]:
                policy_function.reset()

                step_count = 0
                ep_count += 1
                o = env.reset()

                ep_times.append(time.time() - last_time)
                last_time = time.time()

                print("Finish {} episodes with {:.3f} s!".format(ep_count, time.time() - start))
                res = env.get_episode_result()
                res["episode"] = ep_count
                saved_results.append(res)
                df = pd.DataFrame(saved_results)
                print(pretty_print(res))

                path = "evaluate_results/{}{}_backup.csv".format(model_name, suffix)
                print("Backup data is saved at: ", path)
                df.to_csv(path)

                d = {"__all__": False}
                if ep_count >= num_episodes:
                    break
    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    path = "evaluate_results/{}{}.csv".format(model_name, suffix)
    print("Final data is saved at: ", path)
    df.to_csv(path)
    df["model_name"] = model_name
    return df


def get_make_env(env, wrap_with_svo_env=False, render=False):
    if env == "round":

        def make_env(env_id=None):
            return RecorderEnv(MultiAgentRoundaboutEnv(dict(num_agents=40, crash_done=True, use_render=render)))

    elif env == "inter":

        def make_env(env_id=None):
            return RecorderEnv(MultiAgentIntersectionEnv(dict(num_agents=30, crash_done=True, use_render=render)))

    elif env == "parking":

        def make_env(env_id=None):
            return RecorderEnv(
                MultiAgentParkingLotEnv(dict(num_agents=10, crash_done=True, parking_space_num=8, use_render=render))
            )

    elif env == "bottle":

        def make_env(env_id=None):
            return RecorderEnv(MultiAgentBottleneckEnv(dict(num_agents=20, crash_done=True, use_render=render)))

    elif env == "tollgate":

        def make_env(env_id=None):
            from copo.algo_svo.svo_env import get_svo_env
            env_class = MultiAgentTollgateEnv
            if wrap_with_svo_env:
                env_class = get_svo_env(env_class, return_env_class=True)
            return RecorderEnv(env_class(dict(num_agents=40, crash_done=True, use_render=render)))

    else:
        raise ValueError()
    return make_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--use_distributional_svo", action="store_true")
    parser.add_argument("--use_svo_env", action="store_true")
    parser.add_argument("--no_auto_add_svo_to_obs", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    env = args.env
    name = args.name
    num_episodes = 20
    num_checkpoints = 10
    use_distributional_svo = args.use_distributional_svo

    if use_distributional_svo:
        raise ValueError("We find not using use_distributional_svo will be better!")

    suffix = args.suffix

    # for i in range(num_checkpoints):
    model_name = "{}_{}".format(name, env)
    make_env = get_make_env(env, wrap_with_svo_env=args.use_svo_env)
    ret = evaluate_once(
        model_name,
        make_env,
        num_episodes,
        use_distributional_svo=use_distributional_svo,
        suffix=suffix,
        auto_add_svo_to_obs=not args.no_auto_add_svo_to_obs
    )
    if ret is None:
        print("We failed to eval model: ", model_name)
    else:
        print("\n\n\n Finish evaluating model: {}\n\n\n".format(model_name))
