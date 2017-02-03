#!/usr/bin/env python

""" Script to determine whether ParB moves via diffusion or an active mechanism """

import os
import base64
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import itertools

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

def run_filter(data, *args):
    """ *args is a list of tuples containing: (key, operation, value) """
    fil = []
    for key, op, value  in args:
        fil.append(op(data[key], value))
    expr = fil.pop(0)
    for f in fil:
        expr = operator.and_(expr, f)
    return data[expr]


def calculate_diffusion_constant(data):
    pass

def get_stats(x):
    return pd.Series({
        "count": len(x),
        "mean(deltax)": x.deltax.mean(),
        "std(deltax)": x.deltax.std(),
        "mean(deltax)": x.deltax2.mean(),
        "std(deltax2)": x.deltax2.std(),
    })


def determine_diffusion(traces):
    cache_path = os.path.join("ParB_diffusion", ".data", "steps.pandas")
    if os.path.exists(cache_path):
        out_data = pd.read_pickle(cache_path)
        print("Retrieved {0} comparisons from cache".format(len(out_data)))
    else:
        out_data = pd.DataFrame(columns=["pole", "deltat", "deltax", "deltax2"])
        progress_str = "Performed {0:5d} comparisons [{1:3d} of "
        progress_str += "{0:3d} traces]\r".format(len(traces))
        sys.stdout.write(progress_str.format(0, 0))
        sys.stdout.flush()
        total_iterations = 0
        trace_idx = 1
        for data in traces:
            # categorise new or old pole spot based on final position
            last_position = data.iloc[-1]
            if last_position.d_new > last_position.d_old:
                key = "d_old"
                pole = "old"
            else:
                key = "d_new"
                pole = "new"

            for idx1, idx2 in itertools.combinations(data.index.values, 2):
                row1 = data.iloc[idx1]
                row2 = data.iloc[idx2]
                delta_t = row2.timing - row1.timing
                delta_x = row2[key] - row1[key]
                out_data = out_data.append(pd.Series({
                    "pole": pole,
                    "deltat": delta_t,
                    "deltax": delta_x,
                    "deltax2": delta_x ** 2,
                }), ignore_index=True)
                total_iterations += 1
                sys.stdout.write(progress_str.format(total_iterations, trace_idx))
                sys.stdout.flush()
            trace_idx += 1
        pd.to_pickle(out_data, cache_path)
        print()

    fig = plt.figure()

    deltat_cutoff = 220
    new_pole = run_filter(out_data, ("pole", operator.eq, "new"), ("deltat", operator.lt, deltat_cutoff))
    ax = fig.add_subplot(131)
    plot_subset(new_pole, ax, "New pole spots")

    old_pole = run_filter(out_data, ("pole", operator.eq, "old"), ("deltat", operator.lt, deltat_cutoff))
    ax = fig.add_subplot(132, sharex=ax, sharey=ax)
    plot_subset(old_pole, ax, "Old pole spots")

    both_pole = run_filter(out_data, ("deltat", operator.lt, deltat_cutoff))
    ax = fig.add_subplot(133, sharex=ax, sharey=ax)
    plot_subset(both_pole, ax, "All spots")

    fig.tight_layout()
    fig.savefig("ParB_diffusion/ParB_diffusion.pdf")

def plot_subset(data_subset, ax, title="", xlabel=r"$\Delta t$", ylabel=r"$\left < \Delta x^2 \right>$"):
    plt_settings = {
        "x": "deltat",
        "y": "deltax2",
        "x_estimator": np.mean,
        "truncate": True
    }
    sns.regplot(data=data_subset, ax=ax, **plt_settings)
    grad, _ = np.polyfit(data_subset.deltat, data_subset.deltax2, 1)
    diff_coeff = 60 * grad / 2  # um^2.s^{-1}
    ax.plot(
        [0], [0],
        label=r"$D = \SI{{{0:.3f}}}{{\micro\metre\squared\per\second}}$".format(diff_coeff),
        color=(0, 0, 0, 0)
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    sns.despine()


def get_data_from_dir(d):
    print("Retrieving data from {0}".format(d), end=" ")
    d_b64 = base64.b64encode(d.encode("utf8")).decode("utf8")
    cache_path = os.path.abspath(os.path.join(
        "ParB_diffusion",
        ".data",
        "{0}.pandas".format(d_b64)
    ))
    if os.path.exists(cache_path):
        print("[cached]")
        return pd.read_pickle(cache_path)
    else:
        print()

    orig_dir = os.getcwd()
    os.chdir(d)
    cell_traces = parB_velocity.get_traces(spot_filter=lambda x: x > 1)
    pd.to_pickle(cell_traces, cache_path)

    os.chdir(orig_dir)
    return cell_traces

if __name__ == "__main__":
    parB_traces = []
    if not os.path.exists("ParB_diffusion"):
        os.mkdir("ParB_diffusion")
        os.mkdir("ParB_diffusion/.data")

    if not os.path.exists(os.path.join("ParB_diffusion", ".data", "steps.pandas")):
        for p in shared.get_wanted():
            traces = get_data_from_dir(p)
            parB_traces.extend(traces)
            print("Adding {0} traces ({1} in total)".format(len(traces), len(parB_traces)))
    else:
        parB_traces = []

    determine_diffusion(parB_traces)
