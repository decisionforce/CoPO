"""
Usage:

1. Import CCTrainerForMAOurEnvironment, register_cc_model, get_ccppo_env
2. Call get_ccppo_env(env_class) to get the environment.
2. Build config and pay attention to:
    config.env_config.neighbours_distance: default is 10, I think set to 40 might be good.
    config.num_neighbours: default is 4, set to other values are good ablations.
    config.counterfactual: default is False, should varying this to see impact.
4. Call register_cc_model()
"""

import gym
import numpy as np
import ray.rllib.evaluation.postprocessing as rllib_post
import torch
import torch.nn as nn
from copo.algo_ippo.ippo import merge_with_ippo_config
from copo.callbacks import MultiAgentDrivingCallbacks
from copo.ccenv import get_ccenv
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import validate_config_add_multiagent, get_rllib_compatible_env
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, validate_config as ppo_validate_config
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin as TorchKLCoeffMixin, \
    ppo_surrogate_loss as original_ppo_torch_loss
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule, LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import explained_variance, convert_to_torch_tensor
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType

if hasattr(rllib_post, "discount_cumsum"):
    discount = rllib_post.discount_cumsum
else:
    discount = rllib_post.discount

torch, nn = try_import_torch()

CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"

MAPPO_CONFIG = merge_with_ippo_config(
    {
        "real_parameter_sharing": True,
        COUNTERFACTUAL: False,
        "model": {
            "custom_model": "cc_model",
            # "fcnet_activation": "relu",
        },  # We find relu outpeforms tanh!
        "centralized_critic_obs_dim": -1,
        "num_neighbours": 4,
        "framework": "torch",
        "fuse_mode": "mf",  # In ["concat", "mf"]
        "mf_nei_distance": 10,
    }
)


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, ("num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # ===== Here!!! =====
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = model_config["centralized_critic_obs_dim"]
            assert prev_vf_layer_size > 0

            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, out_size=1, initializer=normc_initializer(1.0), activation_fn=None
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else \
            self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return torch.zeros((1, ))

    def central_value_function(self, obs):
        assert self._value_branch is not None
        return torch.reshape(self._value_branch(self._value_branch_separate(obs)), [-1])


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""
    def __init__(self):
        if self.config["framework"] != "torch":
            raise NotImplementedError("Error")
            self.compute_central_vf = make_tf_callable(self.get_session())(self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


def concat_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):
        neighbours = sample_batch.data['infos'][index]["neighbours"]
        for n_count, n_name in enumerate(neighbours):
            if n_count >= policy.config["num_neighbours"]:
                break
            n_count = n_count
            if n_name in other_agent_batches and \
                    (index < len(other_agent_batches[n_name][1][SampleBatch.CUR_OBS])):
                if policy.config[COUNTERFACTUAL]:
                    start = odim + n_count * other_info_dim
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, start: start + odim] = \
                        other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index]
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, start + odim: start + odim + adim] = \
                        other_agent_batches[n_name][1][SampleBatch.ACTIONS][index]
                else:
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, n_count * odim: (n_count + 1) * odim] = \
                        other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index]
    return sample_batch


def mean_field_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations"""
    for index in range(sample_batch.count):
        neighbours = sample_batch.data['infos'][index]["neighbours"]
        neighbours_distance = sample_batch.data['infos'][index]["neighbours_distance"]
        obs_list = []
        act_list = []
        for n_count, (n_name, n_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if n_dist > policy.config["mf_nei_distance"]:
                continue
            if n_name in other_agent_batches and \
                    (index < len(other_agent_batches[n_name][1][SampleBatch.CUR_OBS])):
                obs_list.append(other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index])
                act_list.append(other_agent_batches[n_name][1][SampleBatch.ACTIONS][index])
        if len(obs_list) > 0:
            sample_batch[CENTRALIZED_CRITIC_OBS][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            if policy.config[COUNTERFACTUAL]:
                sample_batch[CENTRALIZED_CRITIC_OBS][index, odim + odim:odim + odim + adim] = np.mean(act_list, axis=0)
    return sample_batch


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    pytorch = policy.config["framework"] == "torch"
    assert pytorch
    _ = sample_batch[SampleBatch.INFOS]  # touch

    # ===== Grab other's observation and actions to compute the per-agent's centralized values =====
    if (pytorch and hasattr(policy, "compute_central_vf")) or (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None

        o = sample_batch[SampleBatch.CUR_OBS]
        odim = sample_batch[SampleBatch.CUR_OBS].shape[1]
        other_info_dim = odim
        adim = sample_batch[SampleBatch.ACTIONS].shape[1]
        if policy.config[COUNTERFACTUAL]:
            other_info_dim += adim

        sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=sample_batch[SampleBatch.CUR_OBS].dtype
        )
        sample_batch[CENTRALIZED_CRITIC_OBS][:, :odim] = sample_batch[SampleBatch.CUR_OBS]

        if policy.config["fuse_mode"] == "concat":
            sample_batch = concat_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim)
        elif policy.config["fuse_mode"] == "mf":
            sample_batch = mean_field_ccppo_process(
                policy, sample_batch, other_agent_batches, odim, adim, other_info_dim
            )
        else:
            raise ValueError("Unknown fuse mode: {}".format(policy.config["fuse_mode"]))

        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(sample_batch[CENTRALIZED_CRITIC_OBS], policy.device)
        ).cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros.
        _ = sample_batch[SampleBatch.INFOS]  # touch
        o = sample_batch[SampleBatch.CUR_OBS]
        sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=o.dtype
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    # ===== Compute the centralized values' advantage =====
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch, last_r, policy.config["gamma"], policy.config["lambda"], use_gae=policy.config["use_gae"]
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    # func = tf_loss if not policy.config["framework"] == "torch" else torch_loss
    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(train_batch[CENTRALIZED_CRITIC_OBS])
    policy._central_value_out = model.value_function()
    loss = original_ppo_torch_loss(policy, model, dist_class, train_batch)
    model.value_function = vf_saved
    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out),
        "value_targets": torch.mean(train_batch[Postprocessing.VALUE_TARGETS]),
        "advantage_mean": torch.mean(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_min": torch.min(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_max": torch.max(train_batch[Postprocessing.ADVANTAGES]),
        "central_value_mean": torch.mean(policy._central_value_out),
        "central_value_min": torch.min(policy._central_value_out),
        "central_value_max": torch.max(policy._central_value_out),
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy._total_loss,
        "policy_loss": policy._mean_policy_loss,
        "vf_loss": policy._mean_vf_loss,
        "kl": policy._mean_kl,
        "entropy": policy._mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
    }


def get_centralized_critic_obs_dim(
    observation_space_shape, action_space_shape, counterfactual, num_neighbours, fuse_mode
):
    """Get the centralized critic"""
    if fuse_mode == "concat":
        pass
    elif fuse_mode == "mf":
        num_neighbours = 1
    else:
        raise ValueError("Unknown fuse mode: ", fuse_mode)
    num_neighbours += 1
    centralized_critic_obs_dim = num_neighbours * observation_space_shape.shape[0]
    if counterfactual:
        # Do not include ego action!
        centralized_critic_obs_dim += (num_neighbours - 1) * action_space_shape.shape[0]
    return centralized_critic_obs_dim


def make_model(policy, obs_space, action_space, config):
    """Overwrite the model config here!"""
    policy.config["exclude_act_dim"] = np.prod(action_space.shape)
    config["model"]["centralized_critic_obs_dim"] = config["centralized_critic_obs_dim"]
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework=config["framework"]
    )


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    return dict()


def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicyForMAOurEnvironment
    else:
        raise ValueError()


def validate_config(config):
    assert config['real_parameter_sharing']

    from ray.tune.registry import _global_registry, ENV_CREATOR
    env_class = _global_registry.get(ENV_CREATOR, config["env"])
    single_env = env_class(config["env_config"])
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    if config["centralized_critic_obs_dim"] == -1:
        config["centralized_critic_obs_dim"] = get_centralized_critic_obs_dim(
            obs_space["agent0"], act_space["agent0"], config["counterfactual"], config["num_neighbours"],
            config["fuse_mode"]
        )

    validate_config_add_multiagent(config, CCPPOTorchPolicyForMAOurEnvironment, ppo_validate_config)


def get_ccppo_env(env_class):
    return get_rllib_compatible_env(get_ccenv(env_class))


CCPPOTorchPolicyForMAOurEnvironment = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicyForMAOurEnvironment",
    get_default_config=lambda: MAPPO_CONFIG,
    make_model=make_model,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    stats_fn=central_vf_stats,
    before_init=setup_torch_mixins,
    mixins=[TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin, CentralizedValueMixin]
)

CCTrainerForMAOurEnvironment = PPOTrainer.with_updates(
    name="CCPPOTrainerForMAOurEnvironment",
    validate_config=validate_config,
    default_config=MAPPO_CONFIG,
    default_policy=CCPPOTorchPolicyForMAOurEnvironment,
    get_policy_class=get_policy_class,
)


def _test():
    # Testing only!
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)
    config = dict(
        env=get_ccppo_env(MultiAgentRoundaboutEnv),
        env_config=dict(
            start_seed=tune.grid_search([5000]),
            num_agents=10,
            crash_done=True,
        ),
        num_sgd_iter=1,
        rollout_fragment_length=200,
        train_batch_size=512,
        sgd_minibatch_size=256,
        num_workers=0,
        **{COUNTERFACTUAL: tune.grid_search([True, False])},
        fuse_mode=tune.grid_search(["concat", "mf"])
        # fuse_mode=tune.grid_search(["mf"])
    )
    results = train(
        CCTrainerForMAOurEnvironment,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=0,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=args.num_gpus,
        num_seeds=1,
        max_failures=0,
        exp_name=exp_name,
        custom_callback=MultiAgentDrivingCallbacks,
        test_mode=True,
        # local_mode=True
    )


def register_cc_model():
    ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)


def _train():
    # Testing only!
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)
    config = dict(
        env=get_ccppo_env(MultiAgentRoundaboutEnv),
        env_config=dict(
            start_seed=tune.grid_search([5000]),
            num_agents=40,
            crash_done=True,
            neighbours_distance=40,
        ),
        **{COUNTERFACTUAL: tune.grid_search([True, False])}
    )
    results = train(
        CCTrainerForMAOurEnvironment,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=0,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=args.num_gpus,
        num_seeds=1,
        max_failures=0,
        exp_name=exp_name,
        custom_callback=MultiAgentDrivingCallbacks,
        test_mode=True,
        # local_mode=True
    )


if __name__ == "__main__":
    _test()
    # _train()
