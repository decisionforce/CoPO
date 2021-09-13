import copy

from copo.utils import validate_config_add_multiagent
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy, validate_config as PPO_valid, DEFAULT_CONFIG as PPO_CONFIG
from ray.tune.utils import merge_dicts

DEFAULT_IPPO_CONFIG = merge_dicts(
    PPO_CONFIG,
    dict(
        rollout_fragment_length=200,
        sgd_minibatch_size=512,
        train_batch_size=1024,
        num_sgd_iter=5,
        lr=3e-4,
        num_workers=4,
        **{"lambda": 0.95},
        num_cpus_per_worker=0.5,
        num_cpus_for_driver=1,
        svo_lr=3e-4,  # Not used
        svo_num_iters=1,  # Not used
        svo_sgd_minibatch_size=None,
        svo_loss_type=0,  # Not used
        nei_adv_type=0,  # Not used
        grad_type=0,
        use_global_value=True,
    )
)

IPPOTrainer = PPOTrainer.with_updates(
    name="IPPO",
    default_config=DEFAULT_IPPO_CONFIG,
    validate_config=lambda c: validate_config_add_multiagent(c, PPOTFPolicy, PPO_valid)
)


def get_ippo_config(new_config):
    org = copy.deepcopy(DEFAULT_IPPO_CONFIG)
    return merge_dicts(org, new_config)


def merge_with_ippo_config(config):
    return get_ippo_config(config)
