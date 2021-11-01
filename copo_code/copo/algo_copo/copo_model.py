import gym
import numpy as np
from copo.algo_copo.constants import *
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.typing import TensorType, ModelConfigDict

tf1, tf, tfv = try_import_tf()


class NeiValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config.get("use_gae"):
            if self.config[USE_CENTRALIZED_CRITIC]:

                @make_tf_callable(self.get_session(), dynamic_shape=True)
                def cc_v(ob, prev_action=None, prev_reward=None, *state):
                    return self.model.value_function(ob)

                self.get_cc_value = cc_v

                @make_tf_callable(self.get_session(), dynamic_shape=True)
                def nei_v(ob, prev_action=None, prev_reward=None, *state):
                    return self.model.get_nei_value(ob)

                self.get_nei_value = nei_v

                @make_tf_callable(self.get_session(), dynamic_shape=True)
                def global_v(ob, prev_action=None, prev_reward=None, *state):
                    return self.model.get_global_value(ob)

                self.get_global_value = global_v

            else:

                @make_tf_callable(self.get_session())
                def nei_value(ob, prev_action, prev_reward, *state):
                    model_out, _ = self.model(
                        {
                            SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                            SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                            SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                            "is_training": tf.convert_to_tensor(False),
                        }, [tf.convert_to_tensor([s]) for s in state], tf.convert_to_tensor([1])
                    )
                    return self.model.get_nei_value()[0]

                @make_tf_callable(self.get_session())
                def global_value(ob, prev_action, prev_reward, *state):
                    model_out, _ = self.model(
                        {
                            SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                            SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                            SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                            "is_training": tf.convert_to_tensor(False),
                        }, [tf.convert_to_tensor([s]) for s in state], tf.convert_to_tensor([1])
                    )
                    return self.model.get_global_value()[0]

                self.get_nei_value = nei_value
                self.get_global_value = global_value

        else:
            raise ValueError()

    def assign_svo(self, svo_param, svo_std_param=None):
        if self.config[USE_DISTRIBUTIONAL_SVO]:
            assert svo_std_param is not None
            return self.get_session().run(
                [self._svo_assign_op, self._svo_std_assign_op],
                feed_dict={
                    self._svo_ph: svo_param,
                    self._svo_std_ph: svo_std_param
                }
            )
        else:
            return self.get_session().run(self._svo_assign_op, feed_dict={self._svo_ph: svo_param})


def register_copo_model():
    ModelCatalog.register_custom_model("copo_model", CoPOModel)


class CoPOModel(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str
    ):
        super(CoPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")
        use_centralized_critic = model_config[USE_CENTRALIZED_CRITIC]
        self.use_centralized_critic = use_centralized_critic

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            raise ValueError()

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)), ), name="observations")
        if use_centralized_critic:
            cc_inputs = tf.keras.layers.Input(
                shape=(model_config["centralized_critic_obs_dim"], ), name="cc_observations"
            )

        # ===== Build Policy Network =====
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # The action distribution outputs.
        logits_out = None
        i = 1
        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size, name="fc_{}".format(i), activation=activation, kernel_initializer=normc_initializer(1.0)
            )(last_layer)
            i += 1
        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            raise ValueError()
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
            if num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs, name="fc_out", activation=None, kernel_initializer=normc_initializer(0.01)
                )(last_layer)
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                raise ValueError()
        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:
            raise ValueError()

        # ===== Build original value function =====
        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            if use_centralized_critic:
                last_vf_layer = cc_inputs
            else:
                last_vf_layer = inputs
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_vf_layer)
                i += 1
        else:
            raise ValueError()
        value_out = tf.keras.layers.Dense(
            1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_vf_layer if last_vf_layer is not None else last_layer)

        if use_centralized_critic:
            self.base_model = tf.keras.Model(inputs, (logits_out if logits_out is not None else last_layer))
            self.cc_value_network = tf.keras.Model(cc_inputs, value_out)
        else:
            self.base_model = tf.keras.Model(
                inputs, [(logits_out if logits_out is not None else last_layer), value_out]
            )
        self._value_out = None

        # ===== Build neighbours value function =====
        if use_centralized_critic:
            last_vf_layer = cc_inputs
        else:
            last_vf_layer = inputs
        i = 1
        for size in hiddens:
            last_vf_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_nei_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_vf_layer)
            i += 1
        value_out_nei = tf.keras.layers.Dense(
            1, name="value_out_nei", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_vf_layer if last_vf_layer is not None else last_layer)
        if use_centralized_critic:
            self.nei_value_network = tf.keras.Model(cc_inputs, value_out_nei)
        else:
            self.nei_value_network = tf.keras.Model(inputs, value_out_nei)
        self._last_nei_value = None

        # ===== Build global value function =====
        if use_centralized_critic:
            last_vf_layer = cc_inputs
        else:
            last_vf_layer = inputs
        i = 1
        for size in hiddens:
            last_vf_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_global_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_vf_layer)
            i += 1
        value_out_global = tf.keras.layers.Dense(
            1, name="value_out_global", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_vf_layer if last_vf_layer is not None else last_layer)
        if use_centralized_critic:
            self.global_value_network = tf.keras.Model(cc_inputs, value_out_global)
        else:
            self.global_value_network = tf.keras.Model(inputs, value_out_global)
        self._last_global_value = None

    def forward(self, input_dict, state, seq_lens):
        if self.use_centralized_critic:
            # Only forward the policy network and left all value functions later.
            model_out = self.base_model(input_dict["obs_flat"])
        else:
            model_out, self._value_out = self.base_model(input_dict["obs_flat"])
            self._last_nei_value = self.nei_value_network(input_dict["obs_flat"])
            self._last_global_value = self.global_value_network(input_dict["obs_flat"])
        return model_out, state

    def value_function(self, cc_obs=None) -> TensorType:
        if self.use_centralized_critic:
            assert cc_obs is not None
            return tf.reshape(self.cc_value_network(cc_obs), [-1])
        else:
            return tf.reshape(self._value_out, [-1])

    def get_nei_value(self, cc_obs=None):
        if self.use_centralized_critic:
            assert cc_obs is not None
            return tf.reshape(self.nei_value_network(cc_obs), [-1])
        else:
            return tf.reshape(self._last_nei_value, [-1])

    def get_global_value(self, cc_obs=None):
        if self.use_centralized_critic:
            assert cc_obs is not None
            return tf.reshape(self.global_value_network(cc_obs), [-1])
        else:
            return tf.reshape(self._last_global_value, [-1])
