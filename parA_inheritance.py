#!/usr/bin/env python

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import seaborn as sns
sns.set_context("paper")
sns.set_style("white")
import os
import pandas as pd
import hashlib
import scipy.stats
from analysis_lib import shared

plt.rc("font", family="sans-serif")
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\usepackage{eulervm}",
    r"\sisetup{detect-all}",
]

PX = 0.12254
DATA_INDEX = [
    "topdir", "subdir", "parent_id",
    "child1_id", "child2_id",
    "parent_lin", "child1_lin", "child2_lin",
    "child_ratio", "max_ratio",
    "length_ratio", "area_ratio",
    "parent_growth", "child1_growth", "child2_growth",
    "parent_elongation", "child1_elongation", "child2_elongation",
    "child1_split", "child2_split",
    "parent_length", "child1_length", "child2_length",
    "parent_area", "child1_area", "child2_area",
    "child1_total", "child2_total",
    "child1_max", "child2_max",
]


def reconfigure_data(data, x, y):
    series1 = data[(data[x] > 1)]
    series2 = data[(data[x] < 1)]
    y1 = pd.concat([series1[y[0]], series2[y[1]]])
    y2 = pd.concat([series1[y[1]], series2[y[0]]])
    ratio = pd.concat([series1[x], 1 / series2[x]])
    indata = {
        "x": ratio,
        "y1": y1,
        "y2": y2
    }
    re_data = pd.DataFrame(indata)
    return re_data


def reconfigure_all_data(data, x):
    swaps = [
        ("child1_id", "child2_id"),
        ("child1_lin", "child2_lin"),
        ("child1_growth", "child2_growth"),
        ("child1_elongation", "child2_elongation"),
        ("child1_split", "child2_split"),
        ("child1_length", "child2_length"),
        ("child1_area", "child2_area"),
        ("child1_total", "child2_total"),
        ("child1_max", "child2_max"),
    ]
    inverts = [
        "child_ratio", "max_ratio",
        "length_ratio", "area_ratio",
    ]

    series1 = data[(data[x] > 1)]
    series2 = data[(data[x] < 1)]

    out_data = {}
    for swap in swaps:
        y1 = pd.concat([series1[swap[0]], series2[swap[1]]])
        y2 = pd.concat([series1[swap[1]], series2[swap[0]]])
        out_data[swap[0]] = y1
        out_data[swap[1]] = y2

    for invert in inverts:
        y = pd.concat([series1[invert], 1 / series2[invert]])
        out_data[invert] = y

    re_data = pd.DataFrame(out_data)
    return re_data


def swarm(ax, data, xlabel1, xlabel2, ylabel):
    sns.swarmplot(data=data[["y1", "y2"]])
    # test that ymax is at least 20% higher than range
    pvalue = scipy.stats.ttest_ind(
        data["y1"].dropna(),
        data["y2"].dropna(),
        equal_var=False
    ).pvalue
    dataset = pd.concat([data["y1"], data["y2"]])
    curr_ylim = ax.get_ylim()
    curr_ymax = curr_ylim[1]
    y20 = dataset.max() + (dataset.max() - dataset.min()) * 0.2
    if curr_ymax < y20:
        ax.set_ylim([curr_ylim[0], y20])
    ymax = dataset.max() + (dataset.max() - dataset.min()) * 0.15
    ax.annotate(
        r"$p = {0:.5f}$".format(pvalue),
        xy=(0.5, ymax),
        horizontalalignment="center",
    )
#    ax.annotate(
#        "",
#        xy=(0, dataset.max()),
#        xytext=(1, dataset.max()),
#        arrowprops={
#            "connectionstyle": "bar",
#            "arrowstyle": "-",
#            "shrinkA": 20,
#            "shrinkB": 20,
#            "lw": 2
#        }
#    )
    ax.set_ylabel(ylabel)
    labels = [xlabel1, xlabel2]
    ax.set_xticklabels(labels)
    sns.despine()


def swarm2(ax, data, xvar1, xlabel1, xvar2, xlabel2, yvar, ylabel):
    sns.swarmplot(data=data[[xvar1, xvar2]])
    pvalue = scipy.stats.ttest_ind(
        data[xvar1].dropna(),
        data[xvar2].dropna(),
        equal_var=False
    ).pvalue
    dataset = pd.concat([data[xvar1], data[xvar2]])
    curr_ylim = ax.get_ylim()
    curr_ymax = curr_ylim[1]
    y20 = dataset.max() + (dataset.max() - dataset.min()) * 0.2
    if curr_ymax < y20:
        ax.set_ylim([curr_ylim[0], y20])
#    ymax = dataset.max() + (dataset.max() - dataset.min()) * 0.15
    ax.annotate(
        r"$p = {0:.5f}$".format(pvalue),
        xy=(0.5, 0.95),
        horizontalalignment="center",
        xycoords=ax.transAxes,
    )
    ax.annotate(
        r"$n = {0}$".format(len(data[xvar1].dropna())),
        xy=(0.5, 0.9),
        horizontalalignment="center",
        xycoords=ax.transAxes,
    )

    ax.set_ylabel(ylabel)
    labels = [xlabel1, xlabel2]
    ax.set_xticklabels(labels)
    sns.despine()


def plot_swarms(data, xvar, xlabels, yvars, title, outfn, redata=False):
    if redata:
        re_data = data
    else:
        re_data = reconfigure_all_data(data, xvar)
    fig = plt.figure(figsize=(5, 12))
    st = fig.suptitle(title, fontsize="x-large")

    plot_num = 1
    for yvar in yvars:
        ax = plt.subplot(3, 2, plot_num)
        yvar1 = "child1_{0}".format(yvar[0])
        yvar2 = "child2_{0}".format(yvar[0])
        swarm2(
            ax, re_data[re_data[xvar] >= 1],
            yvar1, xlabels[0],
            yvar2, xlabels[1],
            "max_ratio", yvar[1]
        )
        plot_num += 1

    plt.tight_layout()

    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)

    plt.savefig("ParA_inheritance/{0}.pdf".format(outfn))
    return re_data


def versus(data, x, y, xlabel="ratio", ylabels=("growth1", "growth2"), outfn="versus"):
    plt.figure(figsize=(12, 6))

    re_data = reconfigure_data(data, x, y)

#    series1 = data[(data[x] > 1)]
#    series2 = data[(data[x] < 1)]
#
#    growth1 = pd.concat([series1[y[0]], series2[y[1]]])
#    growth2 = pd.concat([series1[y[1]], series2[y[0]]])
#    ratio = pd.concat([series1[x], 1 / series2[x]])
#
#    indata = {}
#    indata[xlabel] = ratio
#    indata[ylabels[0]] = growth1
#    indata[ylabels[1]] = growth2
#    re_data = pd.DataFrame(indata)

    plt.subplot(131)
    sns.regplot(x="x", y="y1", data=re_data)
    shared.add_stats(re_data, "x", "y1")
    sns.despine()

    plt.subplot(132)
    sns.regplot(x="x", y="y2", data=re_data)
    shared.add_stats(re_data, "x", "y2")
    sns.despine()

    ax = plt.subplot(133)
    swarm(ax, re_data, ylabels[0], ylabels[1], "growth rate (\si{\per\hour})")

    plt.tight_layout()
    plt.savefig("ParA_inheritance/{0}.pdf".format(outfn))


def get_parB_split(lin, lin_num):
    """ Time until first ParB spot splits, or None if no split """
    parB_paths = shared.get_parB_path(lin, lin[0].T, lin_num)
    lin_t0 = lin[0].T[lin[0].frame - 1]
    split_times = []
    for path in parB_paths:
        if type(path) is shared.TraceConnect:
            if path.split_parent:
                # it has split!
                # get timing of split
                split_times.append(path.timing[0] - lin_t0)
    if not split_times:
        return None
    else:
        return np.min(split_times)


def get_maximal(cell):
    pass


def get_intensity(cell, method="sum"):
    # get mask
    mask_vertices = [cell.mesh[0, 0:2]]
    for coords in cell.mesh[1:-1, 0:2]:
        mask_vertices.append(coords)

    mask_vertices.append(cell.mesh[-1, 0:2])
    for coords in cell.mesh[1:-1, 2:4][::-1]:
        mask_vertices.append(coords)

    mask_vertices.append(cell.mesh[0, 0:2])
    mask_vertices = np.array(mask_vertices)

    x = mask_vertices[:, 0]
    y = mask_vertices[:, 1]
    rows, cols = skimage.draw.polygon(y, x)
    xlim, ylim = cell.parA_img.shape
    rows[rows >= xlim] = xlim - 1
    cols[cols >= ylim] = ylim - 1
    values = cell.parA_img[rows, cols]

    if method == "sum":
        intensity = values.sum()
    else:
        intensity = values.max()

    return intensity


def process():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lookup = json.loads(open("ancestry.json").read())
    siblings = {}  # (mother_lin, daughter_lin, daughter_lin
    cell_lines = {}
    data = pd.DataFrame(columns=DATA_INDEX)

    for l in lin_files:
        c = np.load(l)
        mother_lin = lookup[c[0].id]
        cell_lines[mother_lin] = c
        if c[-1].children:
            siblings[lookup[c[0].id]] = (lookup[c[-1].children[0]], lookup[c[-1].children[1]])

    for parent_num in sorted(siblings.keys()):
        child1_num, child2_num = siblings[parent_num]
#        parent = cell_lines[parent_num][-1]
        # make child1 the smaller cell
        child1 = cell_lines[child1_num][0]
        child2 = cell_lines[child2_num][0]

        if child1.length < child2.length:
            child2_num, child1_num = siblings[parent_num]

        child1 = cell_lines[child1_num][0]
        child2 = cell_lines[child2_num][0]

        parent_lin = cell_lines[parent_num]
        parent_growth = shared.get_growth_rate(parent_lin)
        parent_elong = shared.get_elongation_rate(parent_lin)
        child1_lin = cell_lines[child1_num]
        child1_growth = shared.get_growth_rate(child1_lin)
        child1_elong = shared.get_elongation_rate(child1_lin)
        child2_lin = cell_lines[child2_num]
        child2_growth = shared.get_growth_rate(child2_lin)
        child2_elong = shared.get_elongation_rate(child2_lin)

        c1_inten = get_intensity(child1)
        c2_inten = get_intensity(child2)
        c1_max = get_intensity(child1, "max")
        c2_max = get_intensity(child2, "max")

#        c1_maximal = get_maximal(child1)

        c1_split = get_parB_split(child1_lin, child1_num)
        c2_split = get_parB_split(child2_lin, child2_num)

        if c1_inten == 0:
            continue

        c_ratio = c1_inten / c2_inten  # ratio of intensity between children
        m_ratio = c1_max / c2_max  # ratio of max intensity between children
        l_ratio = (child1.length / child2.length)[0][0]  # ratio of child lengths
        a_ratio = (child1.area / child2.area)[0][0]  # ratio of child areas

        cwd = os.getcwd()
        twd, subdir = os.path.split(cwd)
        topdir = os.path.basename(twd)
        unique_id = hashlib.sha1(
            "{0} {1} {2}".format(topdir, subdir, parent_num).encode("utf-8")
        ).hexdigest()
        temp = [
            topdir,
            subdir,
            cell_lines[parent_num][-1].id,
            child1.id,
            child2.id,
            parent_num,
            child1_num,
            child2_num,
            c_ratio,
            m_ratio,
            l_ratio,
            a_ratio,
            parent_growth, child1_growth, child2_growth,
            parent_elong, child1_elong, child2_elong,
            c1_split, c2_split,
            parent_lin[0].length[0][0] * PX,
            child1_lin[0].length[0][0] * PX,
            child2_lin[0].length[0][0] * PX,
            parent_lin[0].area[0][0] * PX * PX,
            child1_lin[0].area[0][0] * PX * PX,
            child2_lin[0].area[0][0] * PX * PX,
            c1_inten, c2_inten,
            c1_max, c2_max,
        ]
        temp_data = pd.Series(
            data=temp, index=DATA_INDEX, name=unique_id
        )
        data = data.append(temp_data)
    return data


def iterate():
    # iterate through all folders
    original_path = os.getcwd()
    dirs = filter(lambda x: os.path.isdir(x), os.listdir())

    all_data = pd.DataFrame(columns=DATA_INDEX)
    for d in dirs:
        query = input("Process {0}? (Y/n): ".format(d))
        if query.lower() != "n":
            subdirs = filter(lambda y: os.path.isdir(os.path.join(d, y)), os.listdir(d))
            for subdir in subdirs:
                exists = ["mt", "ancestry.json", "lineages.json", "data/cell_lines/lineage01.npy"]
                if sum([os.path.exists(os.path.join(d, subdir, z)) for z in exists]) == len(exists):
                    os.chdir(os.path.join(d, subdir))
                    print("< {0}".format(os.path.join(d, subdir)))
                    out = process()
                    if len(out) > 0:
                        all_data = all_data.append(out)
                    os.chdir(original_path)

    all_data.to_pickle("ParA_inheritance/data.pandas")
    return all_data


def sanity_check(data):
    plt.figure()
    plt.subplot(221)
    plt.title("Total Intensity Ratio")
    sns.despine()
    sns.regplot(x="area_ratio", y="child_ratio", data=data)
    shared.add_stats(data, "area_ratio", "child_ratio")
    plt.xlabel("area\_ratio")
    plt.ylabel("child\_ratio")
    plt.legend(loc=2)

    plt.subplot(222)
    plt.title("Max Intensity Ratio")
    sns.despine()
    sns.regplot(x="area_ratio", y="max_ratio", data=data)
    shared.add_stats(data, "area_ratio", "max_ratio")
    plt.xlabel("area\_ratio")
    plt.ylabel("max\_ratio")
    plt.legend(loc=4)

    plt.subplot(223)
    sns.despine()
    area_rel = data.child_ratio / data.area_ratio
    sns.distplot(area_rel)
    plt.axvline(1, color="k")
    plt.xlabel("child\_ratio / area\_ratio")
    plt.ylabel("Frequency")

    plt.subplot(224)
    sns.despine()
    length_rel = data.max_ratio / data.area_ratio
    sns.distplot(length_rel)
    plt.axvline(1, color="k")
    plt.xlabel("max\_ratio / area\_ratio")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("ParA_inheritance/sanity-check.pdf")


def plot_ratios(data, outfn):
    plt.figure()
    plt.subplot(221)
    plt.title("Total Intensity Ratio")
    sns.despine()
    sns.regplot(x="area_ratio", y="child_ratio", data=data)
    # plot regression line
    shared.add_stats(data, "area_ratio", "child_ratio")
    plt.xlabel("area\_ratio")
    plt.ylabel("child\_ratio")
    plt.legend(loc=2)
#    plt.plot(data.length_ratio, data.child_ratio, ls="none", marker=".")

    plt.subplot(222)
    plt.title("Max Intensity Ratio")
    sns.despine()
    sns.regplot(x="area_ratio", y="max_ratio", data=data)
    shared.add_stats(data, "area_ratio", "max_ratio")
    plt.xlabel("area\_ratio")
    plt.ylabel("max\_ratio")
    plt.legend(loc=4)

    plt.subplot(223)
    sns.despine()
    area_rel = data.child_ratio / data.area_ratio
    sns.distplot(area_rel)
    plt.axvline(1, color="k")
    plt.xlabel("child\_ratio / area\_ratio")
    plt.ylabel("Frequency")

    plt.subplot(224)
    sns.despine()
    length_rel = data.max_ratio / data.area_ratio
    sns.distplot(length_rel)
    plt.axvline(1, color="k")
    plt.xlabel("max\_ratio / area\_ratio")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("ParA_inheritance/{0}.pdf".format(outfn))


def plot_grid(data):
    g = sns.PairGrid(
        data.dropna(),
        vars=[
            "child_ratio", "max_ratio", "area_ratio",
            "parent_growth", "child1_growth", "child2_growth",
        ]
    )
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter)
    g.savefig("ParA_inheritance/all_data.pdf")


def plot_representation(data, spnum):
    areas = pd.concat([data.child1_area, data.child2_area])
    totals = pd.concat([data.child1_total, data.child2_total])
    maxes = pd.concat([data.child1_max, data.child2_max])
    d = pd.DataFrame({"area": areas, "total": totals, "max": maxes})

    plt.subplot(spnum)
    sns.regplot(x="area", y="total", data=d)
    shared.add_stats(d, "area", "total", m=False)
    plt.xlabel("Cell Area (um$^2$)")
    plt.ylabel("Total ParA Intensity (AU)")
    sns.despine()

    plt.subplot(spnum + 1)
    sns.regplot(x="area", y="max", data=d)
    shared.add_stats(d, "area", "max", m=False)
    plt.xlabel("Cell Area (um$^2$)")
    plt.ylabel("Maximum ParA Intensity (AU)")
    sns.despine()


def plot_totals(data, outfn):
    plt.figure()
    plot_representation(data, 221)
    plt.tight_layout()
    plt.savefig("ParA_inheritance/{0}.pdf".format(outfn))


def compare_representations(data1, data2, outfn):
    fig = plt.figure()
    fig.text(0.02, 0.78, r"\textbf{Sorted by length\_ratio}", ha="center", va="center", rotation="vertical")
    plot_representation(data1, 221)

    fig.text(0.02, 0.29, r"\textbf{Sorted by max\_ratio}", ha="center", va="center", rotation="vertical")
    plot_representation(data2, 223)

    plt.tight_layout()

    fig.subplots_adjust(left=0.16)
    plt.savefig("ParA_inheritance/{0}.pdf".format(outfn))


def main():
    if os.path.exists("ParA_inheritance/data.pandas"):
        print("Loading from backup...")
        all_data = pd.read_pickle("ParA_inheritance/data.pandas")
    else:
        all_data = iterate()

    plot_ratios(all_data, "ratios")
#    # plot_grid(all_data)
#
#    versus(
#        all_data, "max_ratio",
#        ("child1_growth", "child2_growth"),
#        "Ratio in maximum ParA intensity",
#        ("High inheritor growth rate", "Low inheritor growth rate"),
#        "max_ratio_growth"
#    )
#
#    versus(
#        all_data, "child_ratio",
#        ("child1_growth", "child2_growth"),
#        "Ratio in total ParA intensity",
#        ("High inheritor growth rate", "Low inheritor growth rate"),
#        "child_ratio_growth"
#    )
#
#    versus(
#        all_data, "area_ratio",
#        ("child1_growth", "child2_growth"),
#        "Ratio in area",
#        ("Larger daughter growth rate", "Smaller daughter growth rate"),
#        "area_ratio_growth"
#    )
#
    default_yvars = [
        ("length", "Length (px)"),
        ("area", r"Area (px$^2$)"),
        ("total", "ParA Total Intensity (AU)"),
        ("max", "ParA Maximum Intensity (AU)"),
        ("elongation", "Elongation Rate (\si{\micro\metre{\per\hour}})"),
        ("growth", "Growth Rate (\si{\per\hour})"),
    ]
    re_data = plot_swarms(
        all_data, "max_ratio", ("High Inheritor", "Low Inheritor"),
        default_yvars, "Maximum ParA Intensity", "max_ratio_swarms"
    )
    plot_swarms(
        all_data, "area_ratio", ("Larger Sibling", "Smaller Sibling"),
        default_yvars, "Cell Area", "area_ratio_swarms"
    )
    plot_swarms(
        all_data, "child_ratio", ("High Inheritor", "Low Inheritor"),
        default_yvars, "Total ParA Intensity", "total_ratio_swarms"
    )

#    compare_representations(all_data, re_data, "total-max-area")
    plot_totals(all_data, "total-max-area")

    plot_ratios(re_data, "ratios-sanity")

if __name__ == "__main__":
    main()
