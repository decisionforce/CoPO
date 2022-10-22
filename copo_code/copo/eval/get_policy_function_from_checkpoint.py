import os
# If pickle raise error, try to use pickle5! Install: pip install pickle5
# and use:
# import pickle5 as pickle
# import pickle

import pandas as pd
from copo.eval.get_policy_function import PolicyFunction, _compute_actions_for_torch_policy2, \
    _compute_actions_for_tf_policy


def get_policy_function_from_checkpoint(algo, ckpt, deterministic=False, policy_name="default"):
    assert os.path.isfile(ckpt), ckpt

    with open(ckpt, "rb") as f:
        data = f.read()

    try:
        import pickle
        unpickled = pickle.loads(data)
        worker = pickle.loads(unpickled.pop("worker"))
    except ValueError or KeyError:
        import pickle5 as pickle
        unpickled = pickle.loads(data)
        worker = pickle.loads(unpickled.pop("worker"))

    if "_optimizer_variables" in worker["state"][policy_name]:
        worker["state"][policy_name].pop("_optimizer_variables")
    weights = worker["state"][policy_name]

    if "copo" in algo:
        layer_name_suffix = "_1"
    else:
        layer_name_suffix = ""

    if "ccppo" in algo:
        weights = {k: v for k, v in weights.items() if "value" not in k}
        policy_class = _compute_actions_for_torch_policy2
    else:
        weights = {k: v for k, v in weights.items() if "value" not in k}
        policy_class = _compute_actions_for_tf_policy

    def policy(obs):
        ret = policy_class(
            weights, obs, policy_name=policy_name, layer_name_suffix=layer_name_suffix, deterministic=deterministic
        )
        return ret

    policy_function = PolicyFunction(policy=policy)
    return policy_function


def get_lcf_from_checkpoint(trial_path):
    file = os.path.join(trial_path, "progress.csv")
    assert os.path.isfile(file), f"We expect to use progress.csv to extract LCF! The folder should be: {trial_path}"
    df = pd.read_csv(file)

    svo_mean = df.loc[df.index[-1], "info/learner/svo"]
    if "info/learner/svo_std" in df:
        svo_std = df.loc[df.index[-1], "info/learner/svo_std"]
    else:
        svo_std = 0.0
    return svo_mean, svo_std
