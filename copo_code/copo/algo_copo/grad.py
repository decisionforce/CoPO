import numpy as np
from copo.algo_copo.constants import *
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf

tf, _, _ = try_import_tf()

NEI_REWARDS = "nei_rewards"
NEI_VALUES = "nei_values"
NEI_ADVANTAGE = "nei_advantage"
NEI_TARGET = "nei_target"
SVO_LR = "svo_lr"

GLOBAL_VALUES = "global_values"
GLOBAL_REWARDS = "global_rewards"
GLOBAL_ADVANTAGES = "global_advantages"
GLOBAL_TARGET = "global_target"


def _flatten(tensor):
    assert tensor is not None
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def build_meta_gradient(policy, optimizer, loss_input_dict, old_model):
    """Implement the idea of gradients bisector to fuse the task gradients
    with the diversity gradient.
    """
    # if not policy.config["grad_type"] != 1:
    #     variables = policy.model.trainable_variables()
    #     policy_grad = optimizer.compute_gradients(loss[0], variables)
    #     if policy.config["grad_clip"] is not None:
    #         clipped_grads, _ = tf.clip_by_global_norm([g for g, _ in policy_grad], policy.config["grad_clip"])
    #         return list(zip(clipped_grads, variables))
    #     else:
    #         return policy_grad

    # Build the loss between new policy and ego advantage.
    logp_ratio = policy._logp_ratio

    if policy.config["use_global_value"]:
        adv = loss_input_dict[GLOBAL_ADVANTAGES]
    else:
        adv = loss_input_dict["normalized_ego_advantages"]

    surrogate_loss = tf.minimum(
        adv * logp_ratio,
        adv * tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"])
    )
    new_policy_ego_loss = tf.reduce_mean(-surrogate_loss)
    new_policy_ego_grad = optimizer.compute_gradients(new_policy_ego_loss, var_list=policy.model.variables())

    # Build the loss between old policy and old log prob.
    old_logits, old_state = old_model(loss_input_dict)
    old_dist = policy.dist_class(old_logits, old_model)
    old_logp = old_dist.logp(loss_input_dict[SampleBatch.ACTIONS])
    old_policy_logp_loss = tf.reduce_mean(old_logp)
    old_policy_logp_grad = optimizer.compute_gradients(old_policy_logp_loss, var_list=policy._old_model.variables())

    # Build the loss between SVO and SVO advantage

    # svo_advantages = tf.cos(svo_rad) * adv + tf.sin(svo_rad) * loss_input_dict[NEI_ADVANTAGE]
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        # Conduct reparameterization trick here!
        svo_rad = tf.random.normal(
            shape=tf.shape(loss_input_dict[NEI_ADVANTAGE]), mean=policy._svo * np.pi / 2, stddev=policy._svo_std
        )
    else:
        svo_rad = policy._svo * np.pi / 2
    advantages = tf.cos(svo_rad) * loss_input_dict[Postprocessing.ADVANTAGES] + \
                 tf.sin(svo_rad) * loss_input_dict[NEI_ADVANTAGE]
    svo_advantages = (advantages - policy._raw_svo_adv_mean) / policy._raw_svo_adv_std

    # with tf.control_dependencies([tf.print(
    #     "GRAD1",
    #         svo_rad, tf.reduce_mean(advantages),
    #         tf.reduce_mean(loss_input_dict[Postprocessing.ADVANTAGES]), tf.reduce_mean(loss_input_dict[NEI_ADVANTAGE]),
    #     "==",
    #         old_policy_logp_loss, new_policy_ego_loss, tf.reduce_mean(svo_advantages)
    # )]):
    #     with tf.control_dependencies([ tf.print(
    #             "GRAD2", policy._raw_svo_adv_mean, policy._raw_svo_adv_std
    #     )]):
    svo_svo_adv_loss = tf.reduce_mean(svo_advantages)
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        svo_var_list = [policy._svo_param, policy._svo_std_param]
    else:
        svo_var_list = [policy._svo_param]
    svo_svo_adv_grad = optimizer.compute_gradients(svo_svo_adv_loss, var_list=svo_var_list)

    # Multiple gradients one by one
    new_policy_ego_grad_flatten = []
    shape_list = []  # For verification used.
    new_policy_ego_grad = [(g, var) for g, var in new_policy_ego_grad if g is not None]
    for g, var in new_policy_ego_grad:
        fg, s, _ = _flatten(g)
        shape_list.append(s)
        new_policy_ego_grad_flatten.append(fg)
    new_policy_ego_grad_flatten = tf.concat(new_policy_ego_grad_flatten, axis=0)
    new_policy_ego_grad_flatten = tf.reshape(new_policy_ego_grad_flatten, (1, -1))

    old_policy_logp_grad_flatten = []
    old_policy_logp_grad = [(g, var) for g, var in old_policy_logp_grad if g is not None]
    for (g, var), verify_shape in zip(old_policy_logp_grad, shape_list):
        fg, s, _ = _flatten(g)
        assert verify_shape == s
        old_policy_logp_grad_flatten.append(fg)
    old_policy_logp_grad_flatten = tf.concat(old_policy_logp_grad_flatten, axis=0)
    old_policy_logp_grad_flatten = tf.reshape(old_policy_logp_grad_flatten, (-1, 1))

    grad_value = tf.matmul(new_policy_ego_grad_flatten, old_policy_logp_grad_flatten)
    final_loss = tf.reshape(grad_value, ()) * svo_svo_adv_loss

    policy._new_policy_ego_loss = new_policy_ego_loss
    policy._old_policy_logp_loss = old_policy_logp_loss
    policy._svo_svo_adv_loss = svo_svo_adv_loss

    single_grad = [g for g, v in svo_svo_adv_grad if g is not None]
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        single_grad_0 = single_grad[0]
        single_grad_1 = single_grad[1]

        final_grad_0 = tf.matmul(grad_value, tf.reshape(single_grad_0, (1, 1)))
        final_grad_0 = tf.reshape(final_grad_0, ())

        final_grad_1 = tf.matmul(grad_value, tf.reshape(single_grad_1, (1, 1)))
        final_grad_1 = tf.reshape(final_grad_1, ())
        return optimizer.apply_gradients(
            [
                [final_grad_0, svo_svo_adv_grad[0][1]],
                [final_grad_1, svo_svo_adv_grad[1][1]],
            ], name="svo_train_op"
        ), final_loss

    else:
        assert len(single_grad) == 1
        single_grad = single_grad[0]

        final_grad = tf.matmul(grad_value, tf.reshape(single_grad, (1, 1)))
        final_grad = tf.reshape(final_grad, ())

        return optimizer.apply_gradients([[final_grad, svo_svo_adv_grad[-1][1]]], name="svo_train_op"), final_loss
