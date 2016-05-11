#!/usr/bin/env python

import glob
import json
import re
import os
import sys
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]

from analysis_lib import shared

PX = 0.12254


def get_growth_rate(delta):
    # exponential method
    t = (delta.timing - delta.timing[0]) / 60
    l = delta.cell_length * PX
    L = np.log(l)
    return np.polyfit(t, L, 1)[0]

def get_elong_rate(delta):
    return np.polyfit(delta.timing / 60, delta.cell_length * PX, 1)[0]

def get_velocity(delta):
    v, _, _, p, _ = scipy.stats.linregress(delta.timing / 60, delta.interspot * PX)
    if p < 0.05:
        return v

def get_delta(one, two):
    spot1 = one.spots()
    spot2 = two.spots()
    timing_range = range(
        spot1["timing"].min(),
        spot1["timing"].max() + 15,
        15
    )
    spot1_dict = dict(zip(one.timing, one.position))
    spot2_dict = dict(zip(two.timing, two.position))

    spot1_distance = []
    spot2_distance = []
    interspot_distance = []
    lengths = []

    for t in timing_range:
        spot1_d = t in spot1_dict and spot1_dict[t] or None
        spot2_d = t in spot2_dict and spot2_dict[t] or None
        spot1_distance.append(spot1_d)
        spot2_distance.append(spot2_d)
        if spot1_d and spot2_d:
            interspot = np.abs(spot1_d - spot2_d)
        else:
            interspot = None
        interspot_distance.append(interspot)
        if spot1_d:
            lengths.append(one.length[one.timing.index(t)])
        elif spot2_d:
            lengths.append(two.length[two.timing.index(t)])
        else:
            lengths.append(None)

    delta = pd.DataFrame(
        {
            "timing": timing_range,
            "spot1_d": spot1_distance,
            "spot2_d": spot2_distance,
            "interspot": interspot_distance,
            "cell_length": lengths,
        }
    ).dropna()

    if len(delta) < 5:
        return pd.DataFrame()

    return delta

def add_parent(p, x):
    s = p.spots()
    x.timing = [s["timing"][-1]] + x.timing
    x.position = [s["position"][-1]] + x.position
    x.intensity = [s["intensity"][-1]] + x.intensity
    x.length = [p.length[-1]] + x.length
    x.spot_ids = [p.spot_ids[-1]] + x.spot_ids

def process():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lineage_nums = sorted([int(re.search("lineage(\d+).npy", x).group(1)) for x in lin_files])
    processable = []
    traces = {}
    for lineage_num in lineage_nums:
        lineage_file = "data/cell_lines/lineage{0:02d}.npy".format(lineage_num)
        cell_line = np.load(lineage_file)
        T = cell_line[0].T
        parB_paths = shared.get_parB_path(cell_line, T, lineage_num)

        for spot_trace in parB_paths:
            if hasattr(spot_trace, "spot_ids"):
                traces[spot_trace.spot_ids[0]] = spot_trace
            if type(spot_trace) is shared.TraceConnect and spot_trace.split_children:
                processable.append((
                    spot_trace.split_children[0],
                    spot_trace.split_children[1],
                    spot_trace
                ))

    velocities = []
    growth_rates = []
    elong_rates = []

    for spot_pair in processable:
        spot1 = traces[spot_pair[0]]
        spot2 = traces[spot_pair[1]]
        parent = spot_pair[2]

        add_parent(parent, spot1)
        add_parent(parent, spot2)

        # take spot with minimum max timing as comparator (spot1)
        if spot1.spots()["timing"].max() > spot2.spots()["timing"].max():
            delta = get_delta(spot2, spot1)
        else:
            delta = get_delta(spot1, spot2)

        if len(delta) >= 5:
            interspot_velocity = get_velocity(delta)
            if interspot_velocity:
                velocities.append(interspot_velocity)
                growth_rates.append(get_growth_rate(delta))
                elong_rates.append(get_elong_rate(delta))

    return velocities, growth_rates, elong_rates
#    return {
#        "velocity": velocities,
#        "growth\_rate": growth_rates,
#        "elong\_rate": elong_rates,
#    }


def plot(data):
    if len(data) == 0:
        return

    plot_order = [
        "delParA", "delParAB",
        "WT ParAB int", "WT ParB int",
        "WT episomal ParB",
    ]
    fig = plt.figure(figsize=(8, 8))

    ax = plt.subplot(221)
    sns.barplot(
        x="dataset", y="v",
        data=data,
        order=plot_order
    )
    _fmt_barplot(ax, r"Mean separation velocity (\si{\micro\metre\per\hour})")

    ax = plt.subplot(222)
    sns.barplot(
        x="dataset", y="elongation",
        data=data,
        order=plot_order
    )
    _fmt_barplot(ax, r"Mean elongation rate (\si{\micro\metre\per\hour})")

    ax = plt.subplot(224)
    sns.barplot(
        x="dataset", y="growth",
        data=data,
        order=plot_order
    )
    _fmt_barplot(ax, r"Mean growth rate (\si{\per\hour})")

    ax = plt.subplot(223)
    sns.countplot(
        x="dataset",
        data=data,
        order=plot_order
    )
    _fmt_barplot(ax, "n")


    plt.tight_layout()
    plt.savefig("parB_interspot/parB_interspot.pdf")

    g = sns.PairGrid(data, vars=["v", "growth", "elongation"], hue="dataset")
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend(bbox_to_anchor=(1.2, 0.55))
    g.savefig("parB_interspot/parB_interspot_data.pdf")


def _fmt_barplot(a, l):
    labels = a.get_xticklabels()
    a.set_xticklabels(labels, rotation=90)
    a.set_xlabel(r"")
    a.set_ylabel(l)
    sns.despine()


if __name__ == "__main__":
    if os.path.exists("mt"):
        data = process()
        plot(*data)
    else:
        if os.path.exists(".tmp/interspot.data"):
            data = pd.read_pickle(".tmp/interspot.data")
            plot(data)
            sys.exit()
        groups = json.loads(open("groupings.json").read())
        orig_dir = os.getcwd()

        velocities = []
        growth_rates = []
        elong_rates = []
        prefixes = []

        for prefix, dirs in groups.items():
            print("* Processing dataset {0}".format(prefix))
            for TLD in dirs:
                for subdir in os.listdir(TLD):
                    target = os.path.join(TLD, subdir)
                    target_conf = [
                        os.path.join(target, "mt", "mt.mat"),
                        os.path.join(target, "data", "cell_lines", "lineage01.npy"),
                    ]
                    conf = [os.path.exists(x) for x in target_conf]
                    if os.path.isdir(target) and sum(conf) == len(conf):
                        os.chdir(target)
                        print("  Handling {0}".format(target))
                        v, g, e = process()
                        velocities.extend(v)
                        growth_rates.extend(g)
                        elong_rates.extend(e)
                        prefixes.extend([prefix] * len(v))

                        os.chdir(orig_dir)

        data = pd.DataFrame({
            "v": velocities,
            "growth": growth_rates,
            "elongation": elong_rates,
            "dataset": prefixes,
        })

        data.to_pickle(".tmp/interspot.data")
        plot(data)
