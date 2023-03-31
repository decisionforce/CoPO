"""
Usage: Call get_svo_env(env_class) to get the real env class!
"""
from collections import defaultdict
from math import cos, sin

import numpy as np
from gym.spaces import Box
from metadrive.envs.marl_envs.marl_tollgate import TollGateObservation, MultiAgentTollgateEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.utils import get_np_random, norm, clip

from copo.utils import get_rllib_compatible_env


class SVOObsForRound(LidarStateObservation):
    @property
    def observation_space(self):
        space = super(SVOObsForRound, self).observation_space
        assert isinstance(space, Box)
        assert len(space.shape) == 1
        length = space.shape[0] + 1
        space = Box(
            low=np.array([space.low[0]] * length),
            high=np.array([space.high[0]] * length),
            shape=(length, ),
            dtype=space.dtype
        )
        return space


class SVOObsForRoundForTollgate(TollGateObservation):
    @property
    def observation_space(self):
        space = super(SVOObsForRoundForTollgate, self).observation_space
        assert isinstance(space, Box)
        assert len(space.shape) == 1
        length = space.shape[0] + 1
        space = Box(
            low=np.array([space.low[0]] * length),
            high=np.array([space.high[0]] * length),
            shape=(length, ),
            dtype=space.dtype
        )
        return space


class SVOEnv:
    @classmethod
    def default_config(cls):
        config = super(SVOEnv, cls).default_config()
        config.update(
            dict(
                neighbours_distance=40,

                # Two mode to compute utility for each vehicle:
                # "linear": util = r_me * svo + r_other * (1 - svo), svo in [0, 1]
                # "angle": util = r_me * cos(svo) + r_other * sin(svo), svo in [0, pi/2]
                # "angle" seems to be more stable!
                svo_mode="angle",
                svo_dist="normal",  # "uniform" or "normal"
                svo_normal_std=0.3,  # The initial STD of normal distribution, might change by calling functions.
                return_native_reward=False,
                include_ego_reward=False,

                # Whether to force set the svo
                force_svo=-100
            )
        )
        return config

    def __init__(self, config=None):
        super(SVOEnv, self).__init__(config)
        self.svo_map = {}
        if hasattr(super(SVOEnv, self), "_update_distance_map"):
            # The parent env might be CCEnv, so we don't need to do this again!
            self._parent_has_distance_map = True
        else:
            self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
            self._parent_has_distance_map = False
        assert self.config["svo_mode"] in ["linear", "angle"]
        assert self.config["svo_dist"] in ["uniform", "normal"]
        assert self.config["svo_normal_std"] > 0.0
        self.force_svo = self.config["force_svo"]

        # Only used in normal SVO distribution
        # SVO is always in range [0, 1], but the real SVO degree is in [-pi/2, pi/2].
        self.current_svo_mean = 0.0  # Set to 0 degree.
        self.current_svo_std = self.config["svo_normal_std"]

    def get_single_observation(self, vehicle_config):
        # TODO we should generalize this function in future!
        if issubclass(self.__class__, MultiAgentTollgateEnv):
            return SVOObsForRoundForTollgate(vehicle_config)
        else:
            return SVOObsForRound(vehicle_config)

    def _get_reset_return(self):
        self.svo_map.clear()
        self._update_distance_map()
        obses = super(SVOEnv, self)._get_reset_return()
        ret = {}
        for k, o in obses.items():
            svo, ret[k] = self._add_svo(o)
            self.svo_map[k] = svo
        return ret

    def step(self, actions):
        # step the environment
        o, r, d, i = super(SVOEnv, self).step(actions)
        self._update_distance_map()

        # add SVO into observation, also update SVO map and info.
        ret = {}
        for k, v in o.items():
            svo, ret[k] = self._add_svo(v, self.svo_map[k] if k in self.svo_map else None, k)
            if k not in self.svo_map:
                self.svo_map[k] = svo
            if i[k]:
                i[k]["svo"] = svo

        if self.config["return_native_reward"]:
            return ret, r, d, i

        # compute the SVO-weighted rewards
        new_rewards = {}

        for k, own_r in r.items():
            other_rewards = []

            if self.config["include_ego_reward"]:
                other_rewards.append(own_r)

            # neighbours = self._find_k_nearest(k, K)
            neighbours = self._find_in_range_for_svo(k, self.config["neighbours_distance"])
            for other_k in neighbours:
                if other_k is None:
                    break
                else:
                    other_rewards.append(r[other_k])
            if len(other_rewards) == 0:
                other_reward = own_r
            else:
                other_reward = np.mean(other_rewards)

            # svo_map stores values in [-1, 1]
            if self.config["svo_mode"] == "linear":
                new_r = self.svo_map[k] * own_r + (1 - self.svo_map[k]) * other_reward
            elif self.config["svo_mode"] == "angle":
                svo = self.svo_map[k] * np.pi / 2
                new_r = cos(svo) * own_r + sin(svo) * other_reward
            else:
                raise ValueError("Unknown SVO mode: {}".format(self.config["svo_mode"]))
            new_rewards[k] = new_r

        return ret, new_rewards, d, i

    def set_force_svo(self, v):
        self.force_svo = v

    def _add_svo(self, o, svo=None, agent_name=None):
        if self.force_svo != -100:
            if self.config["svo_dist"] == "normal":
                svo = get_np_random().normal(loc=self.force_svo, scale=self.current_svo_std)
            else:
                svo = self.force_svo
        elif svo is not None:
            pass
        else:
            if self.config["svo_dist"] == "normal":
                svo = get_np_random().normal(loc=self.current_svo_mean, scale=self.current_svo_std)
                svo = clip(svo, -1, 1)
            else:
                svo = get_np_random().uniform(-1, 1)

            # print("For agent {}, we assign new SVO {} deg! Current mean {} deg, std {}.".format(
            #     agent_name, svo * 90, self.current_svo_mean * 90, self.current_svo_std))

        output_svo = (svo + 1) / 2
        return svo, np.concatenate([o, [output_svo]]).astype(np.float32)

    def set_svo_dist(self, mean, std):
        assert self.config["svo_dist"] == "normal"
        self.current_svo_mean = mean
        self.current_svo_std = std
        assert std > 0.0

    def _find_in_range_for_svo(self, v_id, distance):
        if distance <= 0:
            return []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret

    def _update_distance_map(self):
        if self._parent_has_distance_map:
            return super(SVOEnv, self)._update_distance_map()

        self.distance_map.clear()
        keys = list(self.vehicles.keys())
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = self.vehicles[k1].position
                p2 = self.vehicles[k2].position
                distance = norm(p1[0] - p2[0], p1[1] - p2[1])
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance


def get_svo_env(env_class, return_env_class=False):
    name = env_class.__name__

    class TMP(SVOEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    if return_env_class:
        return TMP
    return get_rllib_compatible_env(TMP)


if __name__ == '__main__':
    # env = SVOEnv({"num_agents": 8, "neighbours_distance": 3, "svo_mode": "angle", "force_svo": 0.9})
    env = get_svo_env(
        MultiAgentTollgateEnv, return_env_class=True
    )({
        "num_agents": 8,
        "neighbours_distance": 3,
        "svo_mode": "angle",
        "svo_dist": "normal"
    })
    o = env.reset()
    assert env.observation_space.contains(o)
    assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [0.0, 1.0] for k in env.vehicles.keys()})
        assert env.observation_space.contains(o)
        assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
        for r_ in r.values():
            total_r += r_
        print("SVO: {}".format({kkk: iii["svo"] if "svo" in iii else None for kkk, iii in info.items()}))
        ep_s += 1
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()
