from typing import Dict, Union, Optional, Type, List

import numpy as np
import ray
from copo.algo_ccppo.ccppo import get_centralized_critic_obs_dim, mean_field_ccppo_process
from copo.algo_copo.constants import *
from copo.algo_copo.copo_model import register_copo_model, NeiValueNetworkMixin
from copo.algo_ippo.ippo import DEFAULT_IPPO_CONFIG, merge_dicts, IPPOTrainer, PPO_valid
from copo.utils import validate_config_add_multiagent
from ray.rllib.agents.ppo.ppo import warn_about_bad_reward_scales, UpdateKL
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, EntropyCoeffSchedule, \
    PPOTFPolicy, LearningRateSchedule
from ray.rllib.evaluation import postprocessing as rllib_post
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.sgd import minibatches, standardized
from ray.rllib.utils.typing import AgentID, TensorType, TrainerConfigDict
from ray.util.iter import LocalIterator

if ray.__version__ == "1.2.0":
    postprocess_ppo_gae = ray.rllib.evaluation.postprocessing.compute_gae_for_sample_batch
else:
    postprocess_ppo_gae = ray.rllib.agents.ppo.ppo_tf_policy.postprocess_ppo_gae

if hasattr(rllib_post, "discount_cumsum"):
    discount = rllib_post.discount_cumsum
else:
    discount = rllib_post.discount
Postprocessing = rllib_post.Postprocessing

# tf, _, _ = try_import_tf()
tf1, tf, tfv = try_import_tf()

DEFAULT_METAPPO_CONFIG = merge_dicts(
    DEFAULT_IPPO_CONFIG,
    {
        "model": {
            "custom_model": "copo_model",
        },
        "initial_svo_std": 0.3,
        USE_CENTRALIZED_CRITIC: False,
        USE_DISTRIBUTIONAL_SVO: False,
        # ===== Details of CC =====
        "centralized_critic_obs_dim": -1,
        "fuse_mode": "mf",
        "counterfactual": True,
        "num_neighbours": -1,  # Not used!
        "mf_nei_distance": 10,
    }
)

register_copo_model()


def compute_nei_advantage(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout[NEI_VALUES], np.array([last_r])])
    delta_t = (rollout[NEI_REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[NEI_ADVANTAGE] = discount(delta_t, gamma * lambda_)
    rollout[NEI_TARGET] = (rollout[NEI_ADVANTAGE] + rollout[NEI_VALUES]).copy().astype(np.float32)
    rollout[NEI_ADVANTAGE] = rollout[NEI_ADVANTAGE].copy().astype(np.float32)
    return rollout


def compute_global_advantage(rollout: SampleBatch, last_r: float, gamma: float = 1.0, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout[GLOBAL_VALUES], np.array([last_r])])
    delta_t = (rollout[GLOBAL_REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[GLOBAL_ADVANTAGES] = discount(delta_t, gamma * lambda_)
    rollout[GLOBAL_TARGET] = (rollout[GLOBAL_ADVANTAGES] + rollout[GLOBAL_VALUES]).copy().astype(np.float32)
    rollout[GLOBAL_ADVANTAGES] = rollout[GLOBAL_ADVANTAGES].copy().astype(np.float32)
    return rollout


def postprocess_add_advantages(policy: Policy, sample_batch: SampleBatch) -> SampleBatch:
    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_global_r = last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])

        if policy.config[USE_CENTRALIZED_CRITIC]:
            last_r = sample_batch[NEI_VALUES][-1]  # This is not accurate, but we just leave it here as approximation.
        else:
            last_r = policy.get_nei_value(
                sample_batch[SampleBatch.NEXT_OBS][-1], sample_batch[SampleBatch.ACTIONS][-1],
                sample_batch[NEI_REWARDS][-1], *next_state
            )

        if policy.config[USE_CENTRALIZED_CRITIC]:
            last_global_r = sample_batch[GLOBAL_VALUES][-1]  # This is not accurate.
        else:
            last_global_r = policy.get_global_value(
                sample_batch[SampleBatch.NEXT_OBS][-1], sample_batch[SampleBatch.ACTIONS][-1],
                sample_batch[GLOBAL_REWARDS][-1], *next_state
            )

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_nei_advantage(sample_batch, last_r, policy.config["gamma"], policy.config["lambda"])

    batch = compute_global_advantage(batch, last_global_r, gamma=1.0, lambda_=policy.config["lambda"])

    return batch


def post_process_fn(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[MultiAgentEpisode] = None
) -> SampleBatch:
    # Put the actions to batch
    infos = sample_batch.get(SampleBatch.INFOS)
    if (infos is not None) and (infos[0] != 0):
        # ===== Fill rewards from data =====
        sample_batch[NEI_REWARDS] = np.array([info["nei_rewards"] for info in infos])
        sample_batch[GLOBAL_REWARDS] = np.array([info["global_rewards"] for info in infos])
        if policy.config[USE_DISTRIBUTIONAL_SVO]:
            sample_batch["step_svo"] = np.array([info["svo"] for info in infos])
        if policy.config[USE_CENTRALIZED_CRITIC]:
            # ===== Fill the centralized observation =====
            o = sample_batch[SampleBatch.CUR_OBS]
            odim = sample_batch[SampleBatch.CUR_OBS].shape[1]
            other_info_dim = odim
            adim = sample_batch[SampleBatch.ACTIONS].shape[1]
            if policy.config[COUNTERFACTUAL]:
                other_info_dim += adim
            sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
                (o.shape[0], policy.config["centralized_critic_obs_dim"]),
                dtype=sample_batch[SampleBatch.CUR_OBS].dtype
            )
            sample_batch[CENTRALIZED_CRITIC_OBS][:, :odim] = sample_batch[SampleBatch.CUR_OBS]
            if policy.config["fuse_mode"] == "mf":
                sample_batch = mean_field_ccppo_process(
                    policy, sample_batch, other_agent_batches, odim, adim, other_info_dim
                )
            else:
                raise ValueError("Unknown fuse mode: {}".format(policy.config["fuse_mode"]))

            # ===== Compute the centralized values =====
            assert SampleBatch.VF_PREDS not in sample_batch
            sample_batch[SampleBatch.VF_PREDS] = policy.get_cc_value(sample_batch[CENTRALIZED_CRITIC_OBS])
            sample_batch[NEI_VALUES] = policy.get_nei_value(sample_batch[CENTRALIZED_CRITIC_OBS])
            sample_batch[GLOBAL_VALUES] = policy.get_global_value(sample_batch[CENTRALIZED_CRITIC_OBS])

    else:
        # ===== Fill the elements if not initialized =====
        sample_batch[NEI_REWARDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch[GLOBAL_REWARDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch["normalized_advantages"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        sample_batch["normalized_ego_advantages"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        _ = sample_batch[SampleBatch.INFOS]  # touch
        o = sample_batch[SampleBatch.CUR_OBS]
        if policy.config[USE_DISTRIBUTIONAL_SVO]:
            sample_batch["step_svo"] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        if policy.config[USE_CENTRALIZED_CRITIC]:
            sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
            sample_batch[NEI_VALUES] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
            sample_batch[GLOBAL_VALUES] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
            sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
                (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=o.dtype
            )

    # ===== Compute the native advantage =====
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]
    sample_batch = compute_advantages(
        sample_batch, last_r, policy.config["gamma"], policy.config["lambda"], use_gae=policy.config["use_gae"]
    )

    # ===== Compute neighbour and global advantage =====
    sample_batch = postprocess_add_advantages(policy, sample_batch)

    # ===== Add some entries =====
    sample_batch["raw_adv"] = sample_batch[Postprocessing.ADVANTAGES]
    sample_batch["raw_nei_adv"] = sample_batch[NEI_ADVANTAGE]
    sample_batch["raw_global_adv"] = sample_batch[GLOBAL_ADVANTAGES]
    return sample_batch


def ppo_lag_surrogate_loss(
    policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution], train_batch: SampleBatch
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Setup the lambda multiplier.
    # penalty_init = 1.0
    # penalty_init = -2.5
    penalty_init = 0.0
    with tf1.variable_scope('svo'):
        # param_init = np.log(max(np.exp(penalty_init) - 1, 1e-8))
        param_init = penalty_init
        svo_param = tf1.get_variable('svo_param', initializer=float(param_init), trainable=True, dtype=tf.float32)

        policy._raw_svo_adv_mean = tf1.placeholder(tf.float32, [], name='raw_svo_adv_mean')
        policy._raw_svo_adv_std = tf1.placeholder(tf.float32, [], name='raw_svo_adv_std')

    # scale svo to [0, 1]
    # svo = tf.clip_by_value((tf.nn.tanh(svo_param) + 1) / 2, 1e-6, 1 - 1e-6)

    # scale svo to [-1, 1]
    svo = tf.clip_by_value(tf.nn.tanh(svo_param), -1 + 1e-6, 1 - 1e-6)

    policy._svo = svo
    policy._svo_param = svo_param
    policy._svo_ph = tf1.placeholder(tf.float32, (), name="svo_ph")
    policy._svo_assign_op = policy._svo_param.assign(policy._svo_ph)

    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        with tf1.variable_scope('svo'):
            # This parameter is log_std!
            svo_std_param = tf1.get_variable(
                'svo_std_param',
                initializer=float(np.log(policy.config["initial_svo_std"])),
                trainable=True,
                dtype=tf.float32
            )
        policy._svo_std = tf.exp(tf.clip_by_value(svo_std_param, -20, 2))
        policy._svo_std_param = svo_std_param
        policy._svo_std_ph = tf1.placeholder(tf.float32, (), name="svo_ph")
        policy._svo_std_assign_op = policy._svo_std_param.assign(policy._svo_std_ph)

    logits, state = model.from_batch(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, mask))

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = tf.reduce_mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logp_ratio = tf.exp(curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP])
    policy._logp_ratio = logp_ratio

    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    _ = train_batch[Postprocessing.ADVANTAGES]  # touch
    _ = train_batch[NEI_ADVANTAGE]  # touch
    _ = train_batch["normalized_ego_advantages"]  # touch
    _ = train_batch[GLOBAL_ADVANTAGES]  # touch
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        _ = train_batch["step_svo"]  # touch

    advantages = train_batch["normalized_advantages"]

    # svo_rad = tf.stop_gradient(policy._svo) * np.pi / 2
    # advantages = tf.cos(svo_rad) * train_batch[Postprocessing.ADVANTAGES] + \
    #              tf.sin(svo_rad) * train_batch[NEI_ADVANTAGE]

    # Normalize advantage to correct scale
    # advantages = advantages / (tf.sin(svo_rad) + tf.sin(svo_rad))

    surrogate_loss = tf.minimum(
        advantages * logp_ratio,
        advantages * tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"])
    )

    mean_policy_loss = reduce_mean_valid(-surrogate_loss)
    # NEI_REWARDS_value_loss = tf.math.square(model.get_nei_value() - train_batch[NEI_TARGET])

    if policy.config[USE_CENTRALIZED_CRITIC]:
        value_fn_out = model.value_function(train_batch[CENTRALIZED_CRITIC_OBS])
        nei_value_fn_out = model.get_nei_value(train_batch[CENTRALIZED_CRITIC_OBS])
        global_value_fn_out = model.get_global_value(train_batch[CENTRALIZED_CRITIC_OBS])
    else:
        value_fn_out = model.value_function()
        nei_value_fn_out = model.get_nei_value()
        global_value_fn_out = model.get_global_value()

    if policy.config["use_gae"]:

        # Original value function loss
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]

        vf_loss1 = tf.math.square(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS])
        vf_clipped = prev_value_fn_out + tf.clip_by_value(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"], policy.config["vf_clip_param"]
        )
        vf_loss2 = tf.math.square(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS])
        vf_loss = tf.maximum(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)

        # Copied neighbour value function loss
        nei_prev_value_fn_out = train_batch[NEI_VALUES]

        nei_vf_loss1 = tf.math.square(nei_value_fn_out - train_batch[NEI_TARGET])
        nei_vf_clipped = nei_prev_value_fn_out + tf.clip_by_value(
            nei_value_fn_out - nei_prev_value_fn_out, -policy.config["vf_clip_param"], policy.config["vf_clip_param"]
        )
        nei_vf_loss2 = tf.math.square(nei_vf_clipped - train_batch[NEI_TARGET])
        nei_vf_loss = tf.maximum(nei_vf_loss1, nei_vf_loss2)
        nei_mean_vf_loss = reduce_mean_valid(nei_vf_loss)

        # Copied global value function loss
        global_prev_value_fn_out = train_batch[GLOBAL_VALUES]

        global_vf_loss1 = tf.math.square(global_value_fn_out - train_batch[GLOBAL_TARGET])
        global_vf_clipped = global_prev_value_fn_out + tf.clip_by_value(
            global_value_fn_out - global_prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"]
        )
        global_vf_loss2 = tf.math.square(global_vf_clipped - train_batch[GLOBAL_TARGET])
        global_vf_loss = tf.maximum(global_vf_loss1, global_vf_loss2)
        global_mean_vf_loss = reduce_mean_valid(global_vf_loss)

        total_loss = (
            -surrogate_loss + policy.kl_coeff * action_kl + policy.config["vf_loss_coeff"] * nei_vf_loss +
            policy.config["vf_loss_coeff"] * global_vf_loss + policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy
        )

        # with tf.control_dependencies([tf.print(
        #         "SHAPE", tf.shape(total_loss), tf.shape(surrogate_loss),
        #         tf.shape(nei_vf_loss), tf.shape(global_vf_loss), tf.shape(vf_loss))]):
        total_loss = reduce_mean_valid(total_loss)

    else:
        raise ValueError()
        mean_vf_loss = tf.constant(0.0)
        total_loss = reduce_mean_valid(
            -surrogate_loss + policy.kl_coeff * action_kl - policy.entropy_coeff * curr_entropy
        )

    # policy._mean_NEI_REWARDS_loss = mean_NEI_REWARDS_loss
    policy._mean_nei_value_loss = nei_mean_vf_loss
    policy._mean_global_mean_vf_loss = global_mean_vf_loss

    # total_loss += penalty_loss  # Do not add it to total loss!
    # total_loss += policy._mean_nei_value_loss

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl

    # policy._svo_loss = reduce_mean_valid(compute_svo_loss(policy, model, dist_class, train_batch))
    policy._adv = reduce_mean_valid(train_batch["raw_adv"])
    policy._adv_nei = reduce_mean_valid(train_batch["raw_nei_adv"])
    policy._adv_global = reduce_mean_valid(train_batch[GLOBAL_ADVANTAGES])
    policy._global_value = reduce_mean_valid(train_batch[GLOBAL_VALUES])
    policy._adv_diff_mean = reduce_mean_valid(train_batch["raw_adv"] - train_batch["raw_nei_adv"])
    policy._adv_diff_max = tf.math.reduce_max(train_batch["raw_adv"] - train_batch["raw_nei_adv"])
    policy._adv_diff_min = tf.math.reduce_min(train_batch["raw_adv"] - train_batch["raw_nei_adv"])
    return total_loss


def new_stats(policy, batch):
    ret = dict(
        {
            "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
            "cur_lr": tf.cast(policy.cur_lr, tf.float64),
            "total_loss": policy._total_loss,
            "policy_loss": policy._mean_policy_loss,
            "vf_loss": policy._mean_vf_loss,
            # "vf_explained_var": explained_variance(
            #     train_batch[Postprocessing.VALUE_TARGETS],
            #     policy.model.value_function()),
            "kl": policy._mean_kl,
            "entropy": policy._mean_entropy,
            "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
        }
    )

    ret["svo"] = policy._svo
    ret["svo_param"] = policy._svo_param
    # ret["nei_policy_loss"] = policy._mean_NEI_REWARDS_loss
    ret["nei_value_loss"] = policy._mean_nei_value_loss
    ret["global_value_loss"] = policy._mean_global_mean_vf_loss
    ret["global_value"] = policy._global_value
    ret["adv"] = policy._adv
    ret["adv_nei"] = policy._adv_nei
    ret["adv_global"] = policy._adv_global
    ret["adv_diff_mean"] = policy._adv_diff_mean
    ret["adv_diff_min"] = policy._adv_diff_min
    ret["adv_diff_max"] = policy._adv_diff_max
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        ret["svo_std"] = policy._svo_std
        ret["svo_std_param"] = policy._svo_std_param
        ret["environment_used_svo"] = tf.reduce_mean(batch["step_svo"])
    return ret


def gradient_fn(policy, optimizer, loss):
    variables = policy.model.trainable_variables()

    assert len([v for v in variables if "penalty_param" in v.name]) == 0
    grads_and_vars = optimizer.compute_gradients(loss, variables)

    # Clip by global norm, if necessary.
    if policy.config["grad_clip"] is not None:
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
        clipped_grads_and_vars = list(zip(policy.grads, variables))
        return clipped_grads_and_vars
    else:
        return grads_and_vars


class UpdateSvo:
    def __init__(self, workers, config):
        self.workers = workers
        self.initialized = False
        self.config = config
        self._svo_local_worker = self.workers.local_worker()
        self._svo_local_policy = self._svo_local_worker.get_policy("default")

    def __call__(self, batch):
        res = self.update_svo(batch)
        metrics = _get_shared_metrics()
        # metrics.info["svo_loss"] = res[1]["svo_loss"]
        metrics.info["current_svo"] = res[1]["svo"]
        metrics.info["current_svo_deg"] = res[1]["svo"] * 90
        print("Current SVO in degree: ", res[1]["svo"] * 90)
        if self._svo_local_policy.config[USE_DISTRIBUTIONAL_SVO]:
            print("Current SVO STD: ", res[1]["svo_std"])
        return batch

    def update_svo(self, data):
        if isinstance(data, tuple):
            batch, fetches = data
        else:
            batch = data
            fetches = {}

        svo_sgd_minibatch_size = self.config["svo_sgd_minibatch_size"]
        if svo_sgd_minibatch_size is None:
            svo_sgd_minibatch_size = self.config["sgd_minibatch_size"]

        svo_param, svo, l1, l2, l3, l4, svo_std_param, svo_std = self.do_minibatch_svo_update(
            batch, self.config["svo_num_iters"], svo_sgd_minibatch_size
        )

        def _update_svo_2(w):
            def _update_svo_1(pi, pi_id):
                pi.assign_svo(svo_param, svo_std_param)
                pi.update_old_policy()

            w.foreach_policy(_update_svo_1)

            if w.get_policy("default").config[USE_DISTRIBUTIONAL_SVO]:

                def _set_env_svo(e):
                    e.set_svo_dist(mean=svo, std=svo_std)

                w.foreach_env(_set_env_svo)
            else:

                def _set_force_svo(e):
                    e.set_force_svo(svo)

                w.foreach_env(_set_force_svo)

        self.workers.foreach_worker(_update_svo_2)

        fetches["svo"] = svo
        fetches["svo_param"] = svo_param
        fetches["raw_svo_adv_mean_value"] = self.workers.local_worker().get_policy("default")._raw_svo_adv_mean_value
        fetches["raw_svo_adv_std_value"] = self.workers.local_worker().get_policy("default")._raw_svo_adv_std_value
        fetches["new_policy_ego_loss"] = l1
        fetches["old_policy_logp_loss"] = l2
        fetches["svo_svo_adv_loss"] = l3
        fetches["svo_final_loss"] = l4
        if self._svo_local_policy.config[USE_DISTRIBUTIONAL_SVO]:
            fetches["svo_std_param"] = svo_std_param
            fetches["svo_std"] = np.exp(svo_std_param)
        return batch, fetches

    def do_minibatch_svo_update(self, batch, num_sgd_iter, sgd_minibatch_size):
        assert len(batch.policy_batches) == 1
        if "is_training" in batch.policy_batches["default"]:
            batch.policy_batches["default"].data.pop("is_training")
        for i in range(num_sgd_iter):
            for minibatch in minibatches(batch.policy_batches["default"], sgd_minibatch_size):
                _, svo, svo_param, l1, l2, l3, l4, svo_std_param, svo_std = self._svo_local_policy.update_svo(minibatch)
        return svo_param, svo, l1, l2, l3, l4, svo_std_param, svo_std


class StandardizeFields:
    def __init__(self, fields: List[str], workers):
        self.fields = fields
        assert len(self.fields) == 2
        self.policy = workers.local_worker().get_policy("default")
        assert self.policy is not None

    def __call__(self, samples):
        wrapped = False

        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)
            wrapped = True

        current_svo = self.policy.get_session().run(self.policy._svo) * np.pi / 2

        for policy_id in samples.policy_batches:
            batch = samples.policy_batches[policy_id]
            if self.policy.config[USE_DISTRIBUTIONAL_SVO]:
                used_svo = batch["step_svo"] * np.pi / 2
            else:
                used_svo = current_svo
            batch["normalized_advantages"] = (
                np.cos(used_svo) * batch[Postprocessing.ADVANTAGES] + np.sin(used_svo) * batch[NEI_ADVANTAGE]
            )

            # Just put the values in policy
            self.policy._raw_svo_adv_mean_value = batch["normalized_advantages"].mean()
            self.policy._raw_svo_adv_std_value = max(1e-4, batch["normalized_advantages"].std())

            batch["normalized_advantages"] = standardized(batch["normalized_advantages"])
            batch["normalized_ego_advantages"] = standardized(batch[Postprocessing.ADVANTAGES])
            batch[GLOBAL_ADVANTAGES] = standardized(batch[GLOBAL_ADVANTAGES])

        if wrapped:
            samples = samples.policy_batches[DEFAULT_POLICY_ID]

        return samples


def execution_plan(workers: WorkerSet, config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the PPO algorithm. Defines the distributed dataflow.

    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(SelectExperiences(workers.trainable_policies()))
    # Concatenate the SampleBatches into one.
    rollouts = rollouts.combine(ConcatBatches(min_batch_size=config["train_batch_size"]))
    # Standardize advantages.
    # <<<<< We add the NEI_REWARDS advantage to normalization too! >>>>>
    # The normalization is conducted in loss update!!!!!!!!!!!!!!!!!!
    rollouts = rollouts.for_each(StandardizeFields(["advantages", NEI_ADVANTAGE], workers))

    # Perform one training step on the combined + standardized batch.
    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"], sgd_minibatch_size=config["sgd_minibatch_size"])
        )
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"],
                framework=config.get("framework")
            )
        )

    # Update SVO
    train_op = train_op.for_each(UpdateSvo(workers, config))
    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    # Warn about bad reward scales and return training metrics.
    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))


def make_model(policy, obs_space, action_space, config):
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    # policy.config["exclude_act_dim"] = np.prod(action_space.shape)
    config["model"]["centralized_critic_obs_dim"] = config["centralized_critic_obs_dim"]
    config["model"][USE_CENTRALIZED_CRITIC] = config[USE_CENTRALIZED_CRITIC]
    if policy.config[USE_CENTRALIZED_CRITIC]:
        assert config["centralized_critic_obs_dim"] != -1

    assert config["model"]["custom_model"]

    policy._old_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework="tf",
        # model_interface=NeiValueNetwork,
        name="copo_old_model"
    )

    new_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework="tf",
        # model_interface=NeiValueNetwork
    )
    return new_model


def setup_mixins_ppo_lag(policy, obs_space, action_space, config):
    # ValueNetworkMixin.__init__(policy, obs_space, action_space, config)  # Don't use native value network
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    NeiValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def vf_preds_fetches(policy: Policy) -> Dict[str, TensorType]:
    if policy.config[USE_CENTRALIZED_CRITIC]:
        return {}
    ret = {SampleBatch.VF_PREDS: policy.model.value_function()}
    if hasattr(policy.model, "get_nei_value"):
        ret[NEI_VALUES] = policy.model.get_nei_value()
    if hasattr(policy.model, "get_global_value"):
        ret[GLOBAL_VALUES] = policy.model.get_global_value()
    return ret


class UpdatePenaltyMixin:
    def __init__(self, o, a, config):
        if config["worker_index"] == 0 and hasattr(self, "_old_model"):
            self._svo_optimizer = tf1.train.AdamOptimizer(learning_rate=self.config[SVO_LR])
            from copo.algo_copo.grad import build_meta_gradient
            self._train_svo_op, self._svo_final_loss = \
                build_meta_gradient(self, self._svo_optimizer, self._loss_input_dict, self._old_model)

            self.model_vars = self.model.variables()
            self.target_model_vars = self._old_model.variables()
            assign_ops = []
            for var, var_target in zip(self.model_vars, self.target_model_vars):
                assign_ops.append(var_target.assign(var))
            self._update_target_expr = tf.group(*assign_ops)
        else:
            self._update_target_expr = None

    def update_svo(self, batch):
        keys = [
            "nei_advantage", "obs", "actions", "advantages", "action_logp", "normalized_ego_advantages",
            GLOBAL_ADVANTAGES
        ]
        feed_dict = {self._loss_input_dict[k]: batch[k] for k in keys}
        feed_dict[self._raw_svo_adv_mean] = self._raw_svo_adv_mean_value
        feed_dict[self._raw_svo_adv_std] = self._raw_svo_adv_std_value
        run_ops = [
            self._train_svo_op, self._svo, self._svo_param, self._new_policy_ego_loss, self._old_policy_logp_loss,
            self._svo_svo_adv_loss, self._svo_final_loss
        ]
        if self.config[USE_DISTRIBUTIONAL_SVO]:
            run_ops.append(self._svo_std_param)
            run_ops.append(self._svo_std)
        ret = self._sess.run(run_ops, feed_dict)
        if not self.config[USE_DISTRIBUTIONAL_SVO]:
            ret = list(ret)
            ret.append(None)
            ret.append(None)
        return ret

    def update_old_policy(self):
        if self._update_target_expr is not None:
            self.get_session().run(self._update_target_expr, feed_dict={})


def after_init(*args, **kwargs):
    UpdatePenaltyMixin.__init__(*args, **kwargs)


def copo_validate(config):
    if config[USE_DISTRIBUTIONAL_SVO]:
        config["env_config"]["return_native_reward"] = True
        config["env_config"]["svo_dist"] = "normal"
        config["env_config"]["svo_normal_std"] = config["initial_svo_std"]

    from ray.tune.registry import _global_registry, ENV_CREATOR
    env_class = _global_registry.get(ENV_CREATOR, config["env"])
    single_env = env_class(config["env_config"])
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    if config[USE_CENTRALIZED_CRITIC] and config["centralized_critic_obs_dim"] == -1:
        config["centralized_critic_obs_dim"] = get_centralized_critic_obs_dim(
            obs_space["agent0"], act_space["agent0"], config["counterfactual"], config["num_neighbours"],
            config["fuse_mode"]
        )

    if config[USE_DISTRIBUTIONAL_SVO]:
        from copo.algo_svo.svo_env import SVOEnv
        assert isinstance(single_env, SVOEnv)
        assert single_env.config["return_native_reward"] is True

    validate_config_add_multiagent(config, CoPOPolicy, PPO_valid)


def optimizer_fn(policy, config):
    return tf1.train.AdamOptimizer(learning_rate=config["lr"])


CoPOPolicy = PPOTFPolicy.with_updates(
    name="CoPOPolicy",
    # get_default_config=lambda: PPO_LAG_CONFIG,
    postprocess_fn=post_process_fn,
    loss_fn=ppo_lag_surrogate_loss,
    gradients_fn=gradient_fn,
    optimizer_fn=optimizer_fn,
    stats_fn=new_stats,
    make_model=make_model,
    before_loss_init=setup_mixins_ppo_lag,
    extra_action_fetches_fn=vf_preds_fetches,
    after_init=after_init,
    # mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin, NeiValueNetworkMixin,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, NeiValueNetworkMixin, UpdatePenaltyMixin]
    # mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin, NeiValueNetworkMixin]
)

CoPO = IPPOTrainer.with_updates(
    name="CoPO",
    default_policy=CoPOPolicy,
    get_policy_class=lambda _: CoPOPolicy,
    execution_plan=execution_plan,
    default_config=DEFAULT_METAPPO_CONFIG,
    validate_config=copo_validate,
    # after_init=after_init,
    # mixins=[UpdatePenaltyMixin]
)
