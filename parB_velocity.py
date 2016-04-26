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
import hashlib
import progressbar

from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]


PX = 0.12254
THRESHOLD = 0  # um / hr threshold for movement
MIN_POINTS = 5


def get_growth_rate(cell_line):
    t = cell_line[0].t
    l = [
        c.length[0][0] * PX for c in cell_line
    ]
    if len(t) < MIN_POINTS:
        return 0

    pf = np.polyfit(t, l, 1)
    growth_rate = pf[0] * 60
    if growth_rate < 0:
        growth_rate = 0

    return growth_rate


def get_traces():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lineage_nums = [int(re.search("lineage(\d+).npy", x).group(1)) for x in lin_files]
    spot_data = []
    progress = progressbar.ProgressBar()
    for lineage_num, lf in progress(list(zip(lineage_nums, lin_files))):
        cell_line = np.load(lf)
        if not hasattr(cell_line[0], "pole_assignment") or not cell_line[0].pole_assignment:
            continue
        pole_assignment = cell_line[0].pole_assignment
        T = cell_line[0].T
        paths = shared.get_parB_path(cell_line, T, lineage_num)
        cell_growth_rate = get_growth_rate(cell_line)
        spot_num = 1
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
            data._spot_num = spot_num
            data._cell_line_id = cell_line[0].id
            data._growth_rate = cell_growth_rate
            data._hash = hashlib.sha256("{0}-{1}-{2}-{3}".format(
                topdir,
                subdir,
                lineage_num,
                spot_num,
            ).encode("utf-8")).hexdigest()
            spot_data.append(data)
            spot_num += 1
    return spot_data


def get_velocities(data):
    velocities = []
    processed = 0
    for spot in data:
        timing = spot.timing
        if len(timing) < MIN_POINTS:
            continue

        vmid = np.polyfit(timing, spot.d_mid, 1)[0] * PX * 60
        # if positive: moving away from new pole
        # if negative: moving towards new pole
        vnew = np.polyfit(timing, spot.d_new, 1)[0] * PX * 60
        # if negative: moving towards new pole
        # if positive: moving away from new pole
        vold = np.polyfit(timing, spot.d_old, 1)[0] * PX * 60
        # if negative: moving towards old pole
        # if positive: moving away from old pole

#        plt.figure()
#        plt.plot(timing, spot.cell_length / 2, "k-", lw=2)
#        plt.plot(timing, -spot.cell_length / 2, "k-", lw=2)
#        plt.plot(timing, spot.d_mid, "r-", label="mid {0}".format(vmid))
#        plt.plot(timing, spot.d_new, "g-", label="new {0}".format(vnew))
#        plt.plot(timing, spot.d_old, "b-", label="old {0}".format(vold))
#        plt.legend()
#        plt.show()
#        plt.close()

        direction = "stationary"
        if spot.d_mid[spot.index[-1]] < 0:
            # closer to new pole
            tether = "new"
            if vnew > THRESHOLD:
                direction = "away"
            elif vnew < -THRESHOLD:
                direction = "towards"
            vabs = np.abs(vnew)
        elif spot.d_mid[spot.index[-1]] > 0:
            tether = "old"
            if vold > THRESHOLD:
                direction = "away"
            elif vold < -THRESHOLD:
                direction = "towards"
            vabs = np.abs(vold)
        else:
            tether = None

#        print("Spot tethered to:", tether)
#        print(spot[["timing", "d_mid"]])
#        print(pd.Series(data={
#            "blah": os.path.join(spot._top_dir, spot._sub_dir),
#            "num": spot._lineage_num,
#            "v_mid": vmid, "v_new": vnew,
#            "v_old": vold, "v_ratio": v_ratio,
#            "growth_rate": spot._growth_rate,
#        }))
#        input("...")

        appendable = (
            spot._path,             # path
            spot._top_dir,          # top_dir
            spot._sub_dir,          # sub_dir
            spot._lineage_num,      # lin
            spot._cell_line_id,     # cid
            spot._hash,             # hash
            vmid,                   # v_mid
            vnew,                   # v_new
            vold,                   # v_old
            np.abs(vabs),           # v_abs
            tether,                 # tether
            direction,              # direction
            spot._growth_rate,      # growth_rate
            len(timing),            # n
        )
        velocities.append(appendable)
        processed += 1

#        old_ax = plt.gca()
#        figsize = np.array(plt.gcf().get_size_inches())
#        figsize[0] = figsize[0] * 2
#        fig = plt.figure(figsize=figsize)
#        _bigax(
#            fig, spec=(1, 1, 1),
#            title=("Designation: nearest pole is {0}, movement is {1}".format(
#                tether, direction
#            ), {"y": 1.08}),
#        )
#        ax = fig.add_subplot(131)
#        plt.title("Relative to mid-cell")
#        plt.xlabel("Time")
#        plt.ylabel("Distance")
#        model = np.polyfit(timing, spot.d_mid, 1)
#        l, = plt.plot(timing, PX * (model[0] * timing + model[1]), "k--")
#        plt.plot(timing, PX * spot.d_mid)
#        plt.plot(timing, PX * spot.cell_length / 2, "k-", lw=2)
#        plt.plot(timing, -PX * spot.cell_length / 2, "k-", lw=2)
#        plt.legend([l], ["v={0:.2f} um/hr".format(vmid)])
#        sns.despine()
#
#        ax = fig.add_subplot(132, sharey=ax)
#        plt.title("Relative to new pole")
#        plt.xlabel("Time")
#        plt.ylabel("Distance")
#        model = np.polyfit(timing, spot.d_new, 1)
#        l, = plt.plot(timing, PX * (model[0] * timing + model[1]), "k--")
#        plt.plot(timing, PX * spot.d_new)
#        plt.legend([l], ["v={0:.2f} um/hr".format(vnew)])
#        sns.despine()
#
#        fig.add_subplot(133, sharey=ax)
#        plt.title("Relative to old pole")
#        plt.xlabel("Time")
#        plt.ylabel("Distance")
#        model = np.polyfit(timing, spot.d_old, 1)
#        l, = plt.plot(timing, PX * (model[0] * timing + model[1]), "k--")
#        plt.plot(timing, PX * spot.d_old)
#        plt.legend([l], ["v={0:.2f} um/hr".format(vold)])
#        sns.despine()
#
#        fn = os.path.join("ParB_velocity", "debug", "{0}.pdf".format(spot._hash))
#        if not os.path.exists("ParB_velocity/debug"):
#            os.makedirs("ParB_velocity/debug")
#        plt.tight_layout()
#        plt.savefig(fn)
#        plt.close()
#        plt.sca(old_ax)

    print("Of which {0} were processed".format(processed))

    indexes = [
        "path", "top_dir", "sub_dir", "lin", "cid", "hash",
        "v_mid", "v_new", "v_old", "v_abs",
        "tether", "direction", "growth_rate", "n",
    ]
    velocities = pd.DataFrame(data=velocities, columns=indexes)
#    print(velocities[["lin", "v_mid", "v_new", "v_old", "tether", "direction", "growth_rate", "n"]])
    return velocities
    # positive = away from new pole
    # negative = towards new pole


def _bigax(fig, xlabel=None, ylabel=None, title=None, spec=(1, 1, 1)):
    big_ax = fig.add_subplot(*spec)
    big_ax.spines["top"].set_color("none")
    big_ax.spines["bottom"].set_color("none")
    big_ax.spines["left"].set_color("none")
    big_ax.spines["right"].set_color("none")
    big_ax.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")

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


def _plot(dataset, **kwargs):
    sns.distplot(dataset, kde=False, **kwargs)
    plt.axvline(lw=0.5, color="k", ls="--", alpha=0.85)
    ax = plt.gca()
    p0 = None
    pn = None
    for p in ax.patches:
        if -THRESHOLD < p.get_x() < THRESHOLD:
            p.set_color("r")
            p0 = p
        else:
            pn = p

    plt.xlabel("")
    return p0, pn


def plot_traces(vdata, prefix=""):
    fig = plt.figure()
    _bigax(
        fig,
        xlabel=("Velocity (\si{\micro\metre\per\hour})", {"labelpad": 10}),
        ylabel=("Frequency", {"labelpad": 25}),
    )

    rows = 2
    cols = 3

    data = [vdata[vdata.tether == "new"], vdata[vdata.tether == "old"]]
    dataselect = ["v_new", "v_old"]
    labels_x = ["All", "Towards", "Away"]
    labels_y = ["New Pole", "Old Pole"]
    i = 1
    ax_c0 = None
    ax_cn = None

    for row_num in range(rows):
        dataset = data[row_num]
        for col_num in range(cols):
            if col_num == 0 and ax_c0:
                ax = fig.add_subplot(rows, cols, i, sharex=ax_c0, sharey=ax_c0)
            elif col_num > 0 and ax_cn:
                ax = fig.add_subplot(rows, cols, i, sharex=ax_cn, sharey=ax_cn)
            else:
                ax = fig.add_subplot(rows, cols, i)
            sns.despine()
            if row_num == 0:
                ax.set_title(labels_x[col_num])

            d = dataset[dataselect[row_num]]  # e.g. dataset.v_new
            if col_num == 0:
                # plot relative to new pole
                p0, pn = _plot(d)
                if p0:
                    p0.set_label("n = {0}".format(
                        len(d[dataset.direction == "stationary"])
                    ))
                pn.set_label("n = {0}".format(
                    len(d) - len(d[dataset.direction == "stationary"])
                ))
                ax.set_ylabel(labels_y[row_num])
            elif col_num == 1:
                d2 = -d[dataset.direction == "towards"]
                _, pn = _plot(d2)
                pn.set_label("n = {0}".format(len(d2)))

            elif col_num == 2:
                # _plot(vdata.abs_rel_velocity[vdata.direction == "away"])
                d2 = d[dataset.direction == "away"]
                _, pn = _plot(d2)
                pn.set_label("n = {0}".format(len(d2)))
            plt.legend()
            if col_num == 0:
                ax_c0 = ax
            else:
                ax_cn = ax
            i += 1

    ax_c0.set_xlim([-2.0, 2.0])
    ax_c0.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])

    ax_cn.set_xlim([0.0, 2.0])
    ax_cn.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])

    plt.tight_layout()
    if not os.path.exists("ParB_velocity"):
        os.mkdir("ParB_velocity")
    fn = os.path.join(
        "ParB_velocity",
        "{2}-velocity-T{0}-N{1}.pdf".format(THRESHOLD, MIN_POINTS, prefix)
    )
    print("Saved file to {0}".format(fn))
    plt.savefig(fn)
    plt.close()


def plot_stats(vdata, prefix=""):
    # stats plot
    fig = plt.figure()

    sp_num = 1
    rows = 1
    cols = 2
    row_num = 0

    _bigax(
        fig,
        xlabel=("Nearest Pole", {"labelpad": 10}),
        spec=(rows, 1, row_num + 1),
    )

    ax = fig.add_subplot(rows, cols, sp_num)
    if THRESHOLD == 0:
        hue_order = ["towards", "away"]
    else:
        hue_order = ["towards", "away", "stationary"]

    sns.barplot(
        x="tether",
        y="v_abs",
        data=vdata[vdata.tether != None],
        hue="direction",
        order=["new", "old"],
        hue_order=hue_order,
        ci=95
    )

    ax.set_xlabel("")
    ax.set_ylabel("Velocity (\si{\micro\metre\per\hour})")

    # get mean elongation rate of population

    sns.despine()

    ax = fig.add_subplot(rows, cols, sp_num + 1)
    sns.countplot(
        x="tether",
        data=vdata,
        hue="direction",
        order=["new", "old"],
        hue_order=hue_order,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Spots")
    sns.despine()

    plt.tight_layout()
    fn = os.path.join(
        "ParB_velocity",
        "{2}-velocity-stats-T{0}-N{1}.pdf".format(THRESHOLD, MIN_POINTS, prefix)
    )
    print("Saved file to {0}".format(fn))
    plt.savefig(fn)
    plt.close()


def plot_examples(data, vdata, prefix=""):
    datastore = {
        x._hash: x for x in data
    }

    num_rows = 4
    ystretch = 4
    if THRESHOLD > 0:
        directions = ["towards", "away", "stationary"]
        xstretch = 4
        num_cols = 6
    else:
        directions = ["towards", "away"]
        xstretch = 8 / 3
        num_cols = 4

    figsize = np.array(plt.gcf().get_size_inches())
    figsize[0] = figsize[0] * xstretch
    figsize[1] = figsize[1] * ystretch
    fig = plt.figure(figsize=figsize)
    _bigax(fig, title=("New Pole", {"y": 1.08}), spec=(1, 2, 1))
    _bigax(fig, title=("Old Pole", {"y": 1.08}), spec=(1, 2, 2))
    _bigax(
        fig,
        xlabel=("Timing (min)", {"labelpad": 10}),
        ylabel=("Distance from mid-cell (um)", {"labelpad": 10}),
        spec=(1, 1, 1)
    )

    num_samples = 3
    sp_num = 1
    for pole in ["new", "old"]:
        for direction in directions:
            population = vdata[
                vdata.direction == direction
            ][
                vdata.tether == pole
            ]
            samples = population.sample(num_samples)
            selected_data = [datastore[x] for x in samples.hash]

            if pole == "new":
                maxv = vdata.values[population.v_new.abs().idxmax()][5]
            elif pole == "old":
                maxv = vdata.values[population.v_old.abs().idxmax()][5]
            selected_data.append(datastore[maxv])

            sel_num = 0
            for selected in selected_data:
                ax = fig.add_subplot(num_rows, num_cols, sp_num + num_cols * sel_num)
                if sel_num == 0:
                    ax.set_title(direction, y=1.08)

                # plot d_mid
                if pole == "new":
                    vlabel = "v={0:.2f}".format(
                        vdata[vdata.hash == selected._hash].v_new.iloc[0]
                    )
                elif pole == "old":
                    vlabel = "v={0:.2f}".format(
                        vdata[vdata.hash == selected._hash].v_old.iloc[0]
                    )

                artist = ax.plot(
                    selected.timing,
                    PX * selected.d_mid,
                    label=vlabel
                )
                ax.plot(selected.timing, PX * selected.cell_length / 2, "k-", lw=2)
                ax.plot(selected.timing, -PX * selected.cell_length / 2, "k-", lw=2)
                plt.legend(artist, [artist[0].get_label()])
                sns.despine()
                sel_num += 1

            sp_num += 1

    fn = os.path.join(
        "ParB_velocity",
        "{2}-examples-T{0}-N{1}.pdf".format(THRESHOLD, MIN_POINTS, prefix)
    )
    print("Saved file to {0}".format(fn))
    plt.savefig(fn)
    plt.close()


def _sub_analyse(x):
    a = x[(x.v_mid > THRESHOLD)]
    b = x[(x.v_mid < -THRESHOLD)]
    c = x[(x.v_mid <= THRESHOLD) & (x.v_mid >= -THRESHOLD)]
    return(a, b, c)


def sub_analysis(vdata, prefix=""):
    """ Sub-analysis of spots which are moving away from a pole
    to determine whether they're moving towards, away from, or static
    relative to mid-cell
    """
    new = vdata[(vdata.direction == "away") & (vdata.tether == "new")]
    # for new pole, towards mid-cell is +ve v_mid
    new_towards, new_away, new_static = _sub_analyse(new)
    new_set = pd.DataFrame({
        "v_mid": np.hstack([
            new_towards.v_mid.values,
            -new_away.v_mid.values,
            np.abs(new_static.v_mid.values),
        ]),
        "direction": (["towards"] * len(new_towards) +
                      ["away"] * len(new_away) +
                      ["static"] * len(new_static)),
    })

    fig = plt.figure()
    _bigax(
        fig,
        xlabel=("Direction relative to mid-cell", {"labelpad": 10}),
        spec=(2, 1, 1),
    )
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("New Pole")
    sns.barplot(
        x="direction",
        y="v_mid",
        data=new_set,
        order=["towards", "away", "static"],
        ci=95
    )
    ax.set_ylabel("Velocity (\si{\micro\metre\per\hour})")
    ax.set_xlabel("")
    sns.despine()

    old = vdata[(vdata.direction == "away") & (vdata.tether == "old")]
    # for old pole, towards mid-cell is -ve v_mid
    old_away, old_towards, old_static = _sub_analyse(old)
    old_set = pd.DataFrame({
        "v_mid": np.hstack([
            old_away.v_mid.values,
            -old_towards.v_mid.values,
            np.abs(old_static.v_mid.values),
        ]),
        "direction": (["away"] * len(old_away) +
                      ["towards"] * len(old_towards) +
                      ["static"] * len(old_static)),
    })

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Old Pole")
    sns.barplot(
        x="direction",
        y="v_mid",
        data=old_set,
        order=["towards", "away", "static"],
        ci=95
    )
    ax.set_ylabel("Velocity (\si{\micro\metre\per\hour})")
    ax.set_xlabel("")
    sns.despine()

#    _bigax(
#        fig,
#        xlabel=("Velocity (\si{\micro\metre\per\hour})", {"labelpad": 10}),
#        title=("Mid-cell", {"y": 1.08}),
#        spec=(2, 1, 2),
#    )
    ax = fig.add_subplot(2, 2, 3)
#    sns.distplot(new.v_mid, kde=False)
#    ax.set_title("New Pole", y=1.08)
#    ax.set_xlabel("")
#    ax.yaxis.set_visible(False)
#    ax.spines["left"].set_color("none")
    sns.countplot(
        x="direction",
        data=new_set,
        order=["towards", "away", "static"],
    )
    sns.despine()

    ax = fig.add_subplot(2, 2, 4)
#    sns.distplot(old.v_mid, kde=False)
#    ax.set_title("Old Pole", y=1.08)
#    ax.set_xlabel("")
#    ax.yaxis.set_visible(False)
#    ax.spines["left"].set_color("none")
    sns.countplot(
        x="direction",
        data=old_set,
        order=["towards", "away", "static"],
    )
    sns.despine()

    plt.tight_layout()

    fn = os.path.join(
        "ParB_velocity",
        "{2}-away-T{0}-N{1}.pdf".format(THRESHOLD, MIN_POINTS, prefix)
    )
    print("Saved file to {0}".format(fn))
    plt.savefig(fn)
    plt.close()


def run(data, prefix=""):
    print("Got all data (n={0})".format(len(data)))
    if data:
        vdata = get_velocities(data)
        plot_traces(vdata, prefix=prefix)
        plot_stats(vdata, prefix=prefix)
        plot_examples(data, vdata, prefix=prefix)
        sub_analysis(vdata, prefix=prefix)
    else:
        print("No data, skipping")


if __name__ == "__main__":
    if os.path.exists("mt"):
        # go go go
        data = get_traces()
        try:
            prefix = sys.argv[1]
        except IndexError:
            prefix = ""
        run(data, prefix=prefix)
    elif "-g" in sys.argv:
        groups = json.loads(open("groupings.json").read())
        orig_dir = os.getcwd()
        for prefix, dirs in groups.items():
            print("* Processing dataset {0}".format(prefix))
            data = []
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
                        data.extend(get_traces())
                        os.chdir(orig_dir)
            run(data, prefix=prefix)
    elif os.path.exists("wanted.json"):
        # get wanted top-level dirs
        TLDs = json.loads(open("wanted.json").read()).keys()
        orig_dir = os.getcwd()
        data = []
        for TLD in TLDs:
            for subdir in os.listdir(TLD):
                target = os.path.join(TLD, subdir)
                target_conf = [
                    os.path.join(TLD, subdir, "mt", "mt.mat"),
                    os.path.join(TLD, subdir, "data", "cell_lines", "lineage01.npy"),
                    os.path.join(TLD, subdir, "poles.json"),
                ]
                conf = [os.path.exists(x) for x in target_conf]
                if os.path.isdir(target) and sum(conf) == len(conf):
                    os.chdir(target)
                    print("Handling {0}".format(target))
                    data.extend(get_traces())
                    os.chdir(orig_dir)
        run(data, prefix="all")
