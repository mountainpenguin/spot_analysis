#!/usr/bin/env python

""" Script to determine whether ParB moves via diffusion or an active mechanism """

import os
import glob
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

from analysis_lib import shared

THRESHOLD_FRAMES = 5


def diffusion_data(data, xkey):
    combinations = itertools.combinations(zip(data["t"], data[xkey]), 2)
    delta = []
    for c0, c1 in combinations:
        delta_t = c1[0] - c0[0]
        delta_x = np.abs(c1[1] - c0[1])
        delta.append((delta_t, delta_x, delta_x ** 2))

    delta = pd.DataFrame(delta, columns=["delta_t", "delta_x", "delta_x2"])
    return delta


def diffusion(spot_data):
    mid = diffusion_data(spot_data, "x_mid")
    new = diffusion_data(spot_data, "x_new")
    old = diffusion_data(spot_data, "x_old")

    plt.figure()
    plt.subplot(131)
    plt.title("Mid")
    sns.regplot(x="delta_t", y="delta_x2", data=mid)
    shared.add_stats(mid, "delta_t", "delta_x2")
    plt.subplot(132)
    plt.title("New")
    sns.regplot(x="delta_t", y="delta_x2", data=new)
    shared.add_stats(new, "delta_t", "delta_x2")
    plt.subplot(133)
    plt.title("Old")
    sns.regplot(x="delta_t", y="delta_x2", data=old)
    shared.add_stats(old, "delta_t", "delta_x2")
    plt.tight_layout()
    plt.show()




def process():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    spots = []
    for lf in lin_files:
        lineage_num = int(re.search("lineage(\d+).npy", lf).group(1))
        cell_line = np.load(lf)
        elongation_rate = shared.get_elongation_rate(cell_line, discard=True)
        if not elongation_rate or len(cell_line) < THRESHOLD_FRAMES:
            continue

        parB_paths = shared.get_parB_path(cell_line, cell_line[0].T, lineage_num)
        spot_num = 1
        for path in parB_paths:
            spot_trace = path.spots()
            lengths = list(path.len())
            idx = 0
            spot_data = {
                "t": [],
                "intensity": [],
                "cell_length": lengths,
                "x_mid": [],
                "x_new": [],
                "x_old": [],
                "pole_known": cell_line[0].pole_assignment,
                "spot_num": spot_num,
                "lineage_num": lineage_num,
            }
            for x in spot_trace:
                l = lengths[idx]
                x_M = x[1]
                x_N = x[1] + (l / 2)
                x_O = l - x_N
                spot_data["t"].append(x[0])
                spot_data["intensity"].append(x[2])
                spot_data["x_mid"].append(x_M)
                spot_data["x_new"].append(x_N)
                spot_data["x_old"].append(x_O)
                idx += 1
            spot_num += 1
            if len(spot_data["t"]) >= THRESHOLD_FRAMES:
                spots.append(spot_data)
                # calculate diffusion parameters
                d_mid, d_new, d_old = diffusion(spot_data)

    s = pd.DataFrame(spots)
    print(s)


if __name__ == "__main__":
    if os.path.exists("mt"):
        data = process()
