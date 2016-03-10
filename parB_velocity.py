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


def get_growth_rate(cell_line):
    t = cell_line[0].t
    l = [
        c.length[0][0] * PX for c in cell_line
    ]
    if len(t) < 2:
        return 0

    pf = np.polyfit(t, l, 1)
    growth_rate = pf[0] * 60
    if growth_rate < 0:
        growth_rate = 0

    return growth_rate

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
        cell_growth_rate = get_growth_rate(cell_line)
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
            path, subdir = os.path.split(os.getcwd())
            topdir = os.path.basename(path)
            data._path = os.getcwd()
            data._top_dir = topdir
            data._sub_dir = subdir
            data._lineage_num = lineage_num
            data._cell_line_id = cell_line[0].id
            data._growth_rate = cell_growth_rate
            spot_data.append(data)
    return spot_data


def get_velocities(data, ref):
    velocities = []
    indexes = ["path", "top_dir", "sub_dir", "lineage_num", "cell_line_id", "velocity", "abs_velocity", "direction", "growth_rate", "rel_velocity", "abs_rel_velocity"]
    processed = 0
    for spot in data:
        # velocity from new_pole
        timing = spot.timing
        if len(timing) < 2:
            continue
        dataset = spot[ref] * PX
        pf = np.polyfit(timing, dataset, 1)
        velocity = pf[0] * 60
        rel_velocity = velocity - spot._growth_rate

        threshold_velocity = 0.15

        if velocity > threshold_velocity:
            direction = "away"
        elif velocity < -threshold_velocity:
            direction = "towards"
        else:
            direction = "stationary"

        velocities.append((
            spot._path,
            spot._top_dir,
            spot._sub_dir,
            spot._lineage_num,
            spot._cell_line_id,
            velocity,
            np.abs(velocity),
            direction,
            spot._growth_rate,
            rel_velocity,
            np.abs(rel_velocity),
        ))
        # velocities.append(pf[0] * 60)
        processed += 1

#        model = pf[0] * timing + pf[1]
#        plt.figure()
#        plt.plot(timing, dataset)
#        plt.plot(timing, model, "k--")
#        plt.show()

    if ref == MID_CELL:
        print("Of which {0} were processed".format(processed))

    velocities = pd.DataFrame(data=velocities, columns=indexes)
    return velocities
    # positive = away from new pole
    # negative = towards new pole


def _bigax(fig, xlabel=None, ylabel=None, title=None, spec=(1, 1, 1)):
    big_ax = fig.add_subplot(*spec)
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    if type(xlabel) is str:
        big_ax.set_xlabel(xlabel)
    elif type(xlabel) is tuple:
        big_ax.set_xlabel(xlabel[0], **xlabel[1])

    if type(ylabel) is str:
        big_ax.set_ylabel(ylabel)
    elif type(ylabel) is tuple:
        big_ax.set_ylabel(ylabel[0], **ylabel[1])

    if type(title) is str:
        big_ax.set_title(title)
    elif type(title) is tuple:
        big_ax.set_title(title[0], **title[1])

    return big_ax


def _plot(dataset):
    sns.distplot(dataset, kde=False)
    plt.xlabel("")


def plot_traces(new, old, mid):
    fig = plt.figure()
    _bigax(
        fig,
        xlabel=("Velocity (um / h)", {"labelpad": 10}),
        ylabel=("Frequency", {"labelpad": 25}),
    )

    rows = 3
    cols = 3

    data = [new, old, mid]
    labels_x = ["All", "Towards", "Away"]
    labels_y = ["New Pole", "Old Pole", "Mid-Cell"]
    i = 1
    ax = None

    for row_num in range(3):
        vdata = data[row_num]
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
                # _plot(vdata.rel_velocity)
                _plot(vdata.velocity)
                ax.set_ylabel(labels_y[row_num])
                ax.set_xlim([-2, 2])
                ax.set_xticks([-2, -1, 0, 1, 2])
            elif col_num == 1:
                # _plot(vdata.abs_rel_velocity[vdata.direction == "towards"])
                _plot(vdata.abs_velocity[vdata.direction == "towards"])
            elif col_num == 2:
                # _plot(vdata.abs_rel_velocity[vdata.direction == "away"])
                _plot(vdata.velocity[vdata.direction == "away"])
            i += 1

    plt.tight_layout()
    plt.savefig("velocity.pdf")
    size = fig.get_size_inches()
    plt.close()

    # stats
    fig = plt.figure(figsize=(size[0], size[1] * 3))

    sp_num = 1
    rows = 3
    cols = 2
    row_num = 0
    datalabel = "mid-cell"

    for row_num, datalabel in zip(range(rows), ["the new pole", "the old pole", "mid-cell"]):
        _bigax(
            fig,
            xlabel=("Direction of movement", {"labelpad": 10}),
            title=("Relative to {0}".format(datalabel), {"y": 1.04}),
            spec=(3, 1, row_num + 1),
        )

        ax = fig.add_subplot(rows, cols, sp_num)
        sns.barplot(x="direction", y="abs_velocity", data=data[row_num], order=["away", "towards"], ci=95)
        # sns.barplot(x="direction", y="abs_rel_velocity", data=data[row_num], order=["away", "towards"], ci=95)
        ax.set_xlabel("")
        ax.set_ylabel("Velocity (um / h)")
        sns.despine()

        ax = fig.add_subplot(rows, cols, sp_num + 1)
        sns.countplot(x="direction", data=data[row_num], order=["away", "towards", "stationary"])
        ax.set_xlabel("")
        ax.set_ylabel("n")
        sns.despine()
        sp_num += 2

    plt.tight_layout()
    plt.savefig("velocity_stats.pdf")


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
