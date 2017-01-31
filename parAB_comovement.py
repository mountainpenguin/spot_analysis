#!/usr/bin/env python

import os
import json

import scipy.stats
import scipy.linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.axes_grid.anchored_artists
import seaborn as sns

from analysis_lib import shared
import parB_velocity

mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]

sns.set_style("white")
sns.set_context("paper")
sns.set_palette([
    (29 / 256, 56 / 256, 105 / 256),
    (94 / 256, 158 / 256, 110 / 256),
    (251 / 256, 245 / 256, 171 / 256)
])

PX = 0.12254


class Stats(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


def determine_comovement(data):
    print("Processing data")
    columns = ["deltat", "deltax", "dxdt", "meanx", "meandnp", "pole"]
    steps = pd.DataFrame()
    for trace in data:
        current_step = 0
        for index, timestep in trace.iterrows():
            if current_step == 0:
                # skip first step
                current_step += 1
            else:
                delta_t = timestep.timing - previous_step.timing
                delta_x = timestep.d_parA - previous_step.d_parA
                mean_x = (timestep.d_parA + previous_step.d_parA) / 2
                mean_d_new_pole = (timestep.d_new + previous_step.d_new) / 2
                mean_cell_length = (timestep.cell_length + previous_step.cell_length) / 2
                if mean_d_new_pole < mean_cell_length / 3:
                    pole_assignment = "new"
                elif mean_d_new_pole > 2 * mean_cell_length / 3:
                    pole_assignment = "old"
                else:
                    pole_assignment = "mid"
                steps = steps.append(pd.Series({
                    "deltat": delta_t,
                    "deltax": delta_x,
                    "dxdt": (delta_x / delta_t) * 15,
                    "meanx": mean_x,
                    "meandnp": mean_d_new_pole,
                    "pole": pole_assignment,
                }), ignore_index=True)
            previous_step = timestep
    return steps

def get_stats(data):
    twotail = 1 - (1 - 95 / 100) / 2
    tstat = scipy.stats.t.ppf(twotail, df=(len(data) - 2))
    A = np.vstack([data.meanx, np.ones(len(data))]).T
    linreg = scipy.linalg.lstsq(A, data.dxdt)
    gradient, intercept = linreg[0]
    sum_y_residuals = np.sum((data.dxdt - data.dxdt.mean()) ** 2)
    Syx = np.sqrt(sum_y_residuals / (len(data) - 2))
    sum_x_residuals = np.sum((data.meanx - data.meanx.mean()) ** 2)
    Sb = Syx / np.sqrt(sum_x_residuals)
    gradient_ci = tstat * Sb

    Sa = Syx * np.sqrt(
        np.sum(data.meanx ** 2) / (len(data) * sum_x_residuals)
    )
    intercept_ci = tstat * Sa
    pearson_r, pearson_p = scipy.stats.pearsonr(data.meanx, data.dxdt)
    print("m = {0:.5f} +/- {1:.5f}".format(gradient, gradient_ci))
    print("c = {0:.5f} +/- {1:.5f}".format(intercept, intercept_ci))
    print("r = {0:.5f} (p={1:.5f})".format(pearson_r, pearson_p))
    print("n = {0}".format(len(data)))

    return Stats(
        gradient=gradient,
        gradient_ci=gradient_ci,
        intercept=intercept,
        intercept_ci=intercept_ci,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        n=len(data)
    )

def plot_comovement(steps):
    print("Plotting data")
    G = []
    for pole in ["new", "mid", "old"]:
        threshold = 0.00
        pole_data = steps[(steps.pole == pole) & (steps.dxdt.abs() > threshold)]
        stats = get_stats(pole_data)

        kind = "reg"
        if kind == "kde":
            kwargs = {"kind": "kde"}
        elif kind == "reg":
            kwargs = {
                "kind": "reg",
                "joint_kws": {
                    "ci": 95,
                    "scatter_kws": {
                        "s": 4,
                        "color": "darkred",
                    },
                },
                "xlim": [-0.5, 100],
                #"ylim": [-3, 3],

            }
        g = sns.jointplot(
            x="meanx",
            y="dxdt",
            data=pole_data,
            **kwargs
        )
        annotation = "m = {0:.3f} $\pm$ {1:.3f}\n".format(
            stats.gradient,
            stats.gradient_ci
        )
        annotation += "c = {0:.3f} $\pm$ {1:.3f}\n".format(
            stats.intercept,
            stats.intercept_ci
        )
        annotation += "r = {0:.3f}, p = {1:.3f}\n".format(
            stats.pearson_r,
            stats.pearson_p
        )
        annotation += "n = {0}".format(len(pole_data))
        g.annotate(
            lambda x, y: (x, y),
            template=annotation,
            loc="upper right",
            fontsize=12
        )
        xlabel = r"Mean distance from ParA (px)"
        # ylabel = r"Movement per timestep ($\Delta x / \Delta t$; px/min)"
        ylabel = r"Movement per timestep ($\Delta x$; px)"
        if pole == "new":
            g.set_axis_labels("", ylabel, fontsize=12)
        elif pole == "mid":
            g.set_axis_labels(xlabel, "", fontsize=12)
        elif pole == "old":
            g.set_axis_labels("", "", fontsize=12)
        G.append(g)

    fig = plt.figure(figsize=(18, 6))
    for g in G:
        for ax in g.fig.axes:
            fig._axstack.add(fig._make_key(ax), ax)

    fig.axes[0].set_position([0.1, 0.08, 0.7, 0.7])  # joint1
    fig.axes[1].set_position([0.1, 0.78, 0.7, 0.1])  # marg_x1
    fig.axes[2].set_position([0.8, 0.08, 0.1, 0.7])  # marg_y1

    fig.axes[3].set_position([1, 0.08, 0.7, 0.7])  # j2
    fig.axes[4].set_position([1, 0.78, 0.7, 0.1])  # x2
    fig.axes[5].set_position([1.7, 0.08, 0.1, 0.7])  # y2

    fig.axes[6].set_position([1.9, 0.08, 0.7, 0.7])  # j3
    fig.axes[7].set_position([1.9, 0.78, 0.7, 0.1])  # x3
    fig.axes[8].set_position([2.6, 0.08, 0.1, 0.7])  # y3
#    x0, y0, width, height

    if not os.path.exists("parAB_comovement"):
        os.mkdir("parAB_comovement")

    print("Saving figure")
    fig.savefig("parAB_comovement/plot.pdf")


if __name__ == "__main__":
    input_wanted = json.loads(open("wanted.json").read())
    top_level = input_wanted.keys()
    orig_dir = os.getcwd()
    data = []
    for directory in top_level:
        for sub_directory in os.listdir(directory):
            target = os.path.join(directory, sub_directory)
            target_conf = [
                os.path.join(directory, sub_directory, "mt", "mt.mat"),
                os.path.join(directory, sub_directory, "data", "cell_lines", "lineage01.npy"),
                os.path.join(directory, sub_directory, "poles.json"),
            ]
            conf = [os.path.exists(x) for x in target_conf]
            if os.path.isdir(target) and sum(conf) == len(conf):
                os.chdir(target)
                print("Retrieving data from {0}".format(target))
                # s = lambda x: x == 3
                s = None
                cell_traces = parB_velocity.get_traces(spot_filter=s)
                data.extend(cell_traces)
                print("Adding {0} traces ({1} in total)".format(len(cell_traces), len(data)))
                os.chdir(orig_dir)

    steps = determine_comovement(data)
    plot_comovement(steps)
