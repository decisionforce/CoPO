import copy
import json
import numbers
import os
import os.path as osp
from json import JSONDecodeError

import numpy as np
import pandas as pd
import scipy
import scipy.interpolate


def _flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def _parse(p):
    dataframe = []
    fn = p.split("/")[-1]
    with open(osp.join(p, "result.json"), "r") as f:
        for l in f:
            data = json.loads(l)
            data = _flatten_dict(data)
            data["file_name"] = fn
            dataframe.append(data)
    dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe.drop(columns=["config/multiagent/policies/default"])
    # print(len(dataframe), list(dataframe.keys()), dataframe["config/multiagent/policies/default"])
    return dataframe


def parse(root):
    """
    Read and form data into a dataframe
    """
    df = []
    paths = [osp.join(root, p) for p in os.listdir(root) if osp.isdir(osp.join(root, p))]
    for pi, p in enumerate(paths):
        print(f"Finish {pi + 1}/{len(paths)} trials.")
        try:
            ret = _parse(p)
        except (FileNotFoundError, JSONDecodeError):
            print("Path {} not found. Continue.".format(p))
            continue
        if ret is not None:
            df.append(ret)
    if not df:
        print("No Data Found!")
        return None
    df = pd.concat(df)
    return df


def smooth(data, num_points=200, interpolate_x="timesteps_total", interpolate_y=None, y_span=1, splitter="file_name"):
    data = data.copy()
    if num_points <= 0:
        return data
    trial_list = [j for i, j in data.groupby(splitter)]
    num_points_ = int(max(len(df) for df in trial_list))
    print("Found {} points, draw {} points.".format(num_points_, num_points))
    num_points = min(num_points, num_points_)
    range_min = min(df[interpolate_x].min() for df in trial_list)
    range_max = max(df[interpolate_x].max() for df in trial_list)
    interpolate_range = np.linspace(range_min, range_max, num_points)
    keys = data.keys()
    new_trial_list = []
    for df in trial_list:
        mask = np.logical_and(df[interpolate_x].min() < interpolate_range, interpolate_range < df[interpolate_x].max())
        mask_rang = interpolate_range[mask]
        if len(df) > 1:
            new_df = {}
            df = df.reset_index(drop=True)
            for k in keys:
                if isinstance(df[k][0], numbers.Number):
                    try:
                        new_df[k] = scipy.interpolate.interp1d(df[interpolate_x], df[k])(mask_rang)
                    except ValueError:
                        continue
                elif isinstance(df[k][0], list):
                    continue
                else:
                    new_df[k] = df[k].unique()[0]
            new_trial_list.append(pd.DataFrame(new_df))
        else:
            new_trial_list.append(df)
    return pd.concat(new_trial_list, ignore_index=True)


if __name__ == '__main__':

    # Read raw data and prepare some intermediate CSV files
    path = "training_results/ccppo_mf/"
    name = "CCPPO (Mean Field)"
    new_path = "ccppo_mf"

    df = parse(path)
    new_df = smooth(df)
    new_df["algo"] = name
    new_df.to_csv("{}.csv".format(new_path))
    print("Finished ", name)
