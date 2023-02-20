"""
Zhenghao: This is a formal script to evaluate trained population from RLLib's generated files directly!

You can specify the path to a "experiment folder" who contains many "trial folders". You need to make in each
"trial folder", a set of checkpoint folders exist and the progress.csv exists too if you are evaluating CoPO population.
"""

import os
import os.path as osp
import re
import copy
import time
import argparse
import json
import numpy as np
import pandas as pd
from copo import pretty_print
from copo.algo_ccppo.ccppo import get_ccppo_env
from copo.algo_svo.svo_env import get_svo_env
from copo.ccenv import get_ccenv
from copo.eval.get_policy_function_from_checkpoint import get_policy_function_from_checkpoint, get_lcf_from_checkpoint
from copo.eval.recoder import RecorderEnv
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv, MultiAgentRoundaboutEnv, MultiAgentTollgateEnv, \
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv, MultiAgentMetaDrive


def get_env(env, should_wrap_copo_env, should_wrap_cc_env, svo_mean=0.0, svo_std=0.0):
    if "Roundabout" in env:
        env_cls = MultiAgentRoundaboutEnv
        env_name = "Round"
    elif "Intersection" in env:
        env_cls = MultiAgentIntersectionEnv
        env_name = "Inter"
    elif "Parking" in env:
        env_cls = MultiAgentParkingLotEnv
        env_name = "Parking"
    elif "Bottle" in env:
        env_cls = MultiAgentBottleneckEnv
        env_name = "Bottle"
    elif "Tollgate" in env:
        env_cls = MultiAgentTollgateEnv
        env_name = "Tollgate"
    elif "MultiAgentMetaDrive" in env:
        env_cls = MultiAgentMetaDrive
        env_name = "PGMap"
    else:
        raise ValueError()

    if should_wrap_copo_env:
        assert should_wrap_cc_env is False
        env_cls = get_svo_env(get_ccenv(env_cls), return_env_class=True)
        env = env_cls({})
        env.set_svo_dist(mean=svo_mean, std=svo_std)

    elif should_wrap_cc_env:
        assert should_wrap_copo_env is False
        env_cls = get_ccenv(env_cls)
        env = env_cls({})

    else:
        env = env_cls({})

    return RecorderEnv(env), env_name


def get_env_and_start_seed(trial_path):
    param_path = os.path.join(trial_path, "params.json")
    assert os.path.isfile(param_path)
    with open(param_path, "r") as f:
        param = json.load(f)

    if "env_config" not in param:
        print(param)
        raise ValueError()

    start_seed = param["env_config"]["start_seed"]
    env_name = param["env"]
    return env_name, start_seed, param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="eval/demo_raw_checkpoints/copo", help="The path ending up with your exp_name."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="How many episodes you want to run for a single checkpoint."
    )
    args = parser.parse_args()

    print("Evaluation begins. The results will be saved at: ", "./evaluate_results/")

    root = args.root
    num_episodes = args.num_episodes

    root = os.path.abspath(root)
    checkpoint_infos = []
    paths = [(osp.join(root, p), p) for p in os.listdir(root) if osp.isdir(osp.join(root, p))]
    for pi, (trial_path, trial_name) in enumerate(paths):
        print(f"Finish {pi + 1}/{len(paths)} trials.")

        raw_env_name, start_seed, ckpt_config = get_env_and_start_seed(trial_path)

        should_wrap_cc_env = "CCPPO" in trial_name
        should_wrap_copo_env = "CoPO" in trial_name

        # Get checkpoint path
        ckpt_paths = []
        for ckpt_path in os.listdir(trial_path):
            if "checkpoint" in ckpt_path:
                ckpt_paths.append((ckpt_path, int(ckpt_path.split("_")[1])))

        # All checkpoints will be evaluated
        ckpt_paths = sorted(ckpt_paths, key=lambda p: p[1])

        for ckpt_path, ckpt_count in ckpt_paths:
            ckpt_file_path = osp.join(root, trial_path, ckpt_path, ckpt_path.replace("_", "-"))
            print(
                f"We will evaluate checkpoint: Algo-{root.split('/')[-1]}, Env-{raw_env_name}, Seed-{start_seed}, "
                f"Ckpt{ckpt_count}"
            )

            algo = root.split('/')[-1]

            ckpt_info = {
                "path": ckpt_file_path,
                "count": ckpt_count,
                "algo": root.split('/')[-1],
                "env": raw_env_name,
                "seed": start_seed,
                "trial": trial_name,
                "trial_path": trial_path,
                "should_wrap_copo_env": should_wrap_copo_env,
                "should_wrap_cc_env": should_wrap_cc_env
            }

            # ckpt_info["config"] = ckpt_config

            checkpoint_infos.append(ckpt_info)
            # checkpoint_infos.append(copy.deepcopy(ckpt_info))

    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    result_name = f"{root.split('/')[-1]}_evaluate_results"

    # print("checkpoint_infos: ", checkpoint_infos)

    for ckpt_count, ckpt_info in enumerate(checkpoint_infos):

        # ckpt_info = copy.deepcopy(ckpt_info)

        assert os.path.isfile(ckpt_info["path"]), ckpt_info
        policy_function = get_policy_function_from_checkpoint(ckpt_info["algo"], ckpt_info["path"])
        if ckpt_info["should_wrap_copo_env"]:
            lcf_mean, lcf_std = get_lcf_from_checkpoint(ckpt_info["trial_path"])
        else:
            lcf_mean = lcf_std = 0.0

        # Setup environment
        env, formal_env_name = get_env(
            env=ckpt_info["env"],
            should_wrap_copo_env=ckpt_info["should_wrap_copo_env"],
            should_wrap_cc_env=ckpt_info["should_wrap_cc_env"],
            svo_mean=lcf_mean,
            svo_std=lcf_std
        )

        print(
            f"\n === Evaluating Algo-{ckpt_info['algo']}_Env-{formal_env_name}_Seed-{ckpt_info['seed']}_Ckpt-{ckpt_info['count']} ==="
        )
        if ckpt_info["should_wrap_copo_env"]:
            print("We are using CoPO environment! The LCF is set to Mean {}, STD {}".format(lcf_mean, lcf_std))

        # Evaluate this checkpoint for sufficient episodes.
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

                # env.render(mode="topdown")

                if step_count % 100 == 0:
                    print(
                        "Evaluating {} {} {}, Num episodes: {}, Num steps in this episode: {} (Ep time {:.2f}, "
                        "Total time {:.2f})".format(
                            ckpt_info["algo"], formal_env_name, ckpt_info["seed"], ep_count, step_count,
                            np.mean(ep_times),
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

                    print("Finish {} episodes with {:.3f} s!\n".format(ep_count, time.time() - start))
                    res = env.get_episode_result()
                    res.update(ckpt_info)
                    res["episode"] = ep_count
                    res["env"] = formal_env_name
                    saved_results.append(res)
                    df = pd.DataFrame(saved_results)
                    print(
                        pretty_print(
                            {f"=== Evaluation Result for Episode {ep_count}/{num_episodes} {result_name}": res}
                        )
                    )

                    path = f"evaluate_results/{result_name}.csv"
                    print("Backup data is saved at: ", path)
                    df.to_csv(path)

                    d = {"__all__": False}
                    if ep_count >= num_episodes:
                        break
        except Exception as e:
            print("Error encountered: ", e)
        finally:
            env.close()

        df = pd.DataFrame(saved_results)
        path = f"evaluate_results/{result_name}.csv"
        print("Final data is saved at: ", path)
        df.to_csv(path)
