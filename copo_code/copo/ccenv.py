from collections import defaultdict

from metadrive.utils import norm


def get_ccenv(env_class):
    class CCEnv(env_class):
        @classmethod
        def default_config(cls):
            config = super(CCEnv, cls).default_config()
            config["neighbours_distance"] = 10
            return config

        def __init__(self, *args, **kwargs):
            env_class.__init__(self, *args, **kwargs)
            self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

        def step(self, actions):
            o, r, d, i = super(CCEnv, self).step(actions)
            self._update_distance_map()
            for kkk in i.keys():
                neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
                i[kkk]["neighbours"] = neighbours
                i[kkk]["neighbours_distance"] = nei_distances
                # i[kkk]["neighbours_distance"] = nei_distances
                nei_rewards = [r[kkkkk] for kkkkk in neighbours]
                if nei_rewards:
                    i[kkk]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
                else:
                    # i[kkk]["nei_rewards"] = r[kkk]
                    i[kkk]["nei_rewards"] = 0.0  # Do not provides neighbour rewards
                i[kkk]["global_rewards"] = sum(r.values()) / len(r.values())
            return o, r, d, i

        def _find_in_range(self, v_id, distance):
            if distance <= 0:
                return []
            max_distance = distance
            dist_to_others = self.distance_map[v_id]
            dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
            ret = [
                dist_to_others_list[i] for i in range(len(dist_to_others_list))
                if dist_to_others[dist_to_others_list[i]] < max_distance
            ]
            ret2 = [
                dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
                if dist_to_others[dist_to_others_list[i]] < max_distance
            ]
            return ret, ret2

        def _update_distance_map(self):
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

    name = env_class.__name__
    name = "CC{}".format(name)
    CCEnv.__name__ = name
    CCEnv.__qualname__ = name
    return CCEnv
