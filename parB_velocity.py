#!/usr/bin/env python

import os
import json
import sys
from analysis_lib import shared
import glob
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
# import scipy.stats

PX = 0.12254
NEW_POLE = "d_new"
OLD_POLE = "d_old"
MID_CELL = "d_mid"


def get_traces():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lineage_nums = []
    cell_lines = []
    for lf in lin_files:
        cell_line = np.load(lf)
        lineage_nums.append(int(re.search("lineage(\d+).npy", lf).group(1)))
        cell_lines.append(cell_line)

    spot_data = []
    for lineage_num, cell_line in zip(lineage_nums, cell_lines):
        pole_assignment = cell_line[0].pole_assignment
        if not pole_assignment:
            continue
        T = cell_line[0].T
        paths = shared.get_parB_path(cell_line, T, lineage_num)
        for path in paths:
            # path.positions: distance from midcell
            spot_trace = path.spots()
            timing = []
            d_mid = []
            intensity = []
            lengths = path.len()
            for x in spot_trace:
                timing.append(x[0])
                d_mid.append(x[1])
                intensity.append(x[2])
            data = pd.DataFrame(
                data={
                    "timing": timing,
                    "d_mid": d_mid,  # negative = closer to new pole
                    "intensity": intensity,
                    "cell_length": lengths,
                },
            )
            data["d_new"] = data.d_mid + (data.cell_length / 2)
            data["d_old"] = data.cell_length - data.d_new
            spot_data.append(data)

    return spot_data


def get_velocities(data, ref):
    velocities = []
    processed = 0
    for spot in data:
        # velocity from new_pole
        timing = spot.timing
        if len(timing) < 2:
            continue
        dataset = spot[ref] * PX
        pf = np.polyfit(timing, dataset, 1)
        velocities.append(pf[0] * 60)
        processed += 1

#        model = pf[0] * timing + pf[1]
#        plt.figure()
#        plt.plot(timing, dataset)
#        plt.plot(timing, model, "k--")
#        plt.show()

    if ref == NEW_POLE:
        print("Of which {0} were processed".format(processed))
    velocities = np.array(velocities)
    returnable = (
        velocities,                   # all spots
        np.abs(velocities),             # absolute velocities
        velocities[velocities >= 0],  # moving away from new pole
        velocities[velocities <= 0],  # moving towards new pole
    )
    return returnable
    # positive = away from new pole
    # negative = towards new pole


def _plot(dataset):
    sns.distplot(dataset, kde=False)


def plot_traces(new, old, mid):
    fig = plt.figure()

    rows = 3
    cols = 3

    big_ax = fig.add_subplot(111)
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    data = [new, old, mid]
    labels_x = ["All", "Towards", "Away"]
    labels_y = ["New Pole", "Old Pole", "Mid-Cell"]
    i = 1
    ax = None

    for row_num in range(3):
        v_all, v_abs, v_to, v_away = data[row_num]
        for col_num in range(3):
            if ax:
                ax = fig.add_subplot(rows, cols, i, sharey=ax)
            else:
                ax = fig.add_subplot(rows, cols, i)
            sns.despine()
            # lim = np.ceil(v_abs.max())
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 100])
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            if row_num == 0:
                ax.set_title(labels_x[col_num])

            if col_num == 0:
                _plot(v_all)
                # _plot(v_abs)
                ax.set_ylabel(labels_y[row_num])
                ax.set_xlim([-2, 2])
                ax.set_xticks([-2, -1, 0, 1, 2])
            elif col_num == 1:
                _plot(v_to)
            elif col_num == 2:
                _plot(-v_away)
            i += 1

    big_ax.set_xlabel("Velocity (um / h)", labelpad=10)
    big_ax.set_ylabel("Frequency", labelpad=25)

    plt.tight_layout()
    plt.savefig("velocity.pdf")


if __name__ == "__main__":
    if os.path.exists("mt"):
        # go go go
        data = get_traces()
    elif os.path.exists("wanted.json"):
        # get wanted top-level dirs
        TLDs = json.loads(open("wanted.json").read()).keys()
        orig_dir = os.getcwd()
        data = []
        for TLD in TLDs:
            for subdir in os.listdir(TLD):
                target = os.path.join(TLD, subdir)
                target_mt = os.path.join(TLD, subdir, "mt", "mt.mat")
                target_conf = [
                    os.path.join(TLD, subdir, "mt", "mt.mat"),
                    os.path.join(TLD, subdir, "data", "cell_lines", "lineage01.npy"),
                ]
                conf = [os.path.exists(x) for x in target_conf]
                if os.path.isdir(target) and sum(conf) == len(conf):
                    os.chdir(target)
                    print("Handling {0}".format(target))
                    data.extend(get_traces())
                    os.chdir(orig_dir)
    else:
        print("Nothing to do, exiting")
        sys.exit(1)

    print("Got all data (n={0})".format(len(data)))
    new_pole_data = get_velocities(data, NEW_POLE)
    old_pole_data = get_velocities(data, OLD_POLE)
    mid_cell_data = get_velocities(data, MID_CELL)
    plot_traces(new_pole_data, old_pole_data, mid_cell_data)
