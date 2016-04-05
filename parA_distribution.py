#!/usr/bin/env python

"""
    - Take cells from all lineages in child directories
    - Filter out cells that can be observed to divide *twice*
    - Split average cell cycle length in approx. 10 segments
    - Bin cells at each approx. segment
    - Normalise cell length within each bin
    - Plot heatmaps of sum of ParA intensity for each cell cycle bin
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import os
import sys
import json
sns.set_style("white")

n_bins = 10


def get_cell_lines():
    # for now, just deal with one dir at a time
    lin_files = glob.glob("data/cell_lines/lineage*.npy")
    cell_lines = []
    for lf in lin_files:
        cell_line = np.load(lf)
        cell_lines.append(cell_line)
    print("Got {0} cell lines ({1})".format(len(cell_lines), os.getcwd()))
    return cell_lines


def filter_cell_lines(cell_lines, criterion=None):
    out = []
    for cell_line in cell_lines:
        first = cell_line[0]
        last = cell_line[-1]
        if first.parent and last.children and len(cell_line) >= n_bins:
            out.append(cell_line)
    print("Of which {0} are suitable for analysis (have both a parent and children)".format(len(out)))
    return out, remainder(cell_lines)


def remainder(cell_lines):
    firsts = []
    lasts = []
    for cell_line in cell_lines:
        first = cell_line[0]
        last = cell_line[-1]
        if not first.parent and first.pole_assignment and last.children:
            lasts.append(last)

        if not last.children and last.pole_assignment and first.parent:
            firsts.append(first)

    print("There are an extra {0} cells which have no children "
          "which can be added to the first bin".format(len(firsts)))
    print("And an extra {0} cells which have known polarity but no parent "
          "which can be added to the last bin".format(len(lasts)))

    return firsts, lasts


def bin_cells(cell_lines, extra_firsts, extra_lasts):
    """ Bin cells according to cell cycle time """
    n_len_bins = 30  # minimum value

    bins = dict(zip(
        range(n_bins),
        [[] for x in range(n_bins)]
    ))
    # bin 0 is *always* the first cell
    # bins 1 -> (n - 2) are variable
    # bin (n - 1) is *always* the last cell
    for cell_line in cell_lines:
        bins[0].append(cell_line[0])
        bins[n_bins - 1].append(cell_line[-1])

        per_bin = (len(cell_line) - 2) / (n_bins - 2)

        bin_count = 0
        bin_num = 1
        cell_idx = 0
        max_cells = len(cell_line) - 1
        for cell in cell_line:
            segments = len(cell.mesh)
            if segments > n_len_bins:
                n_len_bins = int(segments)

            if cell_idx > 0 and cell_idx < max_cells:
                if bin_count >= per_bin:
                    bin_num += 1
                    bin_count = bin_count - per_bin
                if bin_num >= n_bins - 1:
                    bin_num -= 1
                bins[bin_num].append(cell)
                bin_count += 1
            cell_idx += 1

    bins[0].extend(extra_firsts)
    bins[n_bins - 1].extend(extra_lasts)

    print("# cells in each bin:")
    for k, v in bins.items():
        print(" {0}: {1}".format(k, len(v)))

    return bins, n_len_bins


def resample_bins(cell, n_len_bins):
    new_vals = np.zeros(n_len_bins)
    new_bins = np.array(range(n_len_bins))
    old_vals = cell.parA_fluorescence_unsmoothed
    old_bins = np.array(range(len(old_vals)))
    bin_width = len(old_bins) / (n_len_bins + 1)
    new_bin_idx = 0
    old_range = np.array([0, bin_width])
    if len(old_bins) > n_len_bins:
        # histogram is shrinking
        while new_bin_idx < n_len_bins:
            if old_range[0] != int(old_range[0]):
                old1 = int(np.ceil(old_range[0]))
                pre_rem = old1 - old_range[0]
            else:
                old1 = int(old_range[0])
                pre_rem = 0

            if old_range[1] != int(old_range[1]):
                old2 = int(np.floor(old_range[1]))
                post_rem = old_range[1] - old2
            else:
                old2 = int(old_range[1])
                post_rem = 0

            pre_val = old_vals[old1 - 1] * pre_rem
            if old1 == old2:
                mid_val = 0
            else:
                mid_val = old_vals[old1]
            try:
                post_val = old_vals[old2] * post_rem
            except IndexError:
                post_val = 0

            new_val = pre_val + mid_val + post_val
            new_val /= bin_width
            new_vals[new_bin_idx] = new_val

            old_range += bin_width
            new_bin_idx += 1

    elif len(old_bins) < n_len_bins:
        while new_bin_idx < n_len_bins:
            if old_range[0] != int(old_range[0]):
                old1 = int(np.floor(old_range[0]))
            else:
                old1 = int(old_range[0])

            if old_range[1] != int(old_range[1]):
                old2 = int(np.floor(old_range[1]))
            else:
                old2 = int(old_range[1])

            if old1 == old2:
                new_val = bin_width * old_vals[old1]
            else:
                pre_val = (old2 - old_range[0]) * old_vals[old1]
                try:
                    post_val = (old_range[1] - old2) * old_vals[old2]
                except IndexError:
                    print("IndexError")
                    post_val = 0
                new_val = pre_val + post_val
            new_val /= bin_width
            new_vals[new_bin_idx] = new_val

            old_range += bin_width
            new_bin_idx += 1
    else:
        new_vals = old_vals

    return new_vals, new_bins


def normalise_bin(cells, n_len_bins):
    """ normalise ParA by cell length """
    # split cell in 30 segments
    # return ParA values as % of cell length
    # normalise by intensity
    normalised = None
    for cell in cells:
        rescaled_vals, rescaled_bins = resample_bins(cell, n_len_bins)
        rescaled_vals = rescaled_vals / rescaled_vals.max()
        if normalised is None:
            normalised = rescaled_vals
        else:
            normalised += rescaled_vals
    normalised = normalised / normalised.max()
    return normalised


def plot_bins(bins, n_len_bins, cycle_bins, ax=None):
    cmapper = plt.cm.get_cmap("afmhot")
    vals = np.array([x for x in bins.values()])
    max_inten = vals.max()
    if not ax:
        plt.figure()
        ax_top = plt.subplot2grid((30, 14), (0, 0), colspan=12, rowspan=2)
        ax_top.xaxis.set_visible(False)
        ax_top.yaxis.set_visible(False)
        sns.despine(bottom=True, left=True, ax=ax_top)

        ax_main = plt.subplot2grid((30, 14), (2, 0), colspan=12, rowspan=12)
        ax_main.set_xlim([0, 100])
        ax_main.set_ylim([0, 100])
        ax_main.set_xlabel("% of cell cycle")
        ax_main.set_ylabel("% distance from new pole")
        sns.despine(ax=ax_main)

        ax_key = plt.subplot2grid((30, 14), (2, 13), colspan=1, rowspan=12)
        ax_key.xaxis.set_ticks_position("none")
        ax_key.yaxis.tick_right()
        ax_key.yaxis.set_ticks_position("none")
        ax_key.set_xticks([])
        ax_key.set_ylabel("Intensity Key")
        plt.sca(ax_key)
        plt.yticks([0, 128, 255], ["100%", "50%", "0%"])

        ax_other = plt.subplot2grid((30, 14), (18, 0), colspan=12, rowspan=12)
        ax_other.set_xlabel("% distance from new pole")
        ax_other.set_ylabel("Normalised intensity (%)")

        sns.despine(ax=ax_other)

    data = np.array([
        (
            (x[0] / n_bins) * 100,  # xval
            len(x[1]),  # yval
        ) for x in cycle_bins.items()
    ])
    bar_width = 100 / n_bins
    bars = ax_top.bar(
        data[:, 0],
        data[:, 1],
        bar_width
    )

    # label bars with n values
    for bar in bars:
        height = bar.get_height()
        ax_top.text(
            bar.get_x() + (bar.get_width() / 2),
            1.05 * height,
            "{0:d}".format(int(height)),
            ha="center",
            va="bottom"
        )

    for bin_num, normalised in bins.items():
        colours = cmapper(normalised / max_inten)
        i = 0
        for norm in normalised:
            x_pos = (bin_num / (n_bins)) * 100
            y_pos = (i / (n_len_bins - 1)) * 100  # % distance from new pole
            r = matplotlib.patches.Rectangle(
                (x_pos, y_pos),
                width=100 / n_bins,
                height=100 / (n_len_bins - 1),
                facecolor=colours[i],
                edgecolor="none"
            )
            ax_main.add_patch(r)
            i += 1

        if bin_num == 0:
            label = "Birth"
            colour = "k"
        elif bin_num == (n_bins - 1):
            label = "Division"
            colour = "r"
        else:
            colour = None
        xvals = (np.arange(n_len_bins) / (n_len_bins - 1)) * 100
        yvals = 100 * normalised / max_inten
        if colour:
            ax_other.plot(xvals, yvals, alpha=1, label=label, color=colour)
        else:
            ax_other.plot(xvals, yvals, color=(bin_num/n_bins, 0, 0), alpha=.6, label=bin_num)
    ax_other.legend(bbox_to_anchor=(1.3, 1.1))

    # plot colour bar key
    g = np.array([np.linspace(0, 1, 256)[::-1]]).T
#    g = np.vstack((g, g)).T

    ax_key.imshow(g, aspect="auto", cmap=cmapper)

    if not os.path.exists("ParA_distribution"):
        os.mkdir("ParA_distribution")

    fn = os.path.join(
        "ParA_distribution",
        "ParA_distribution-C{0}-L{1}.pdf".format(n_bins, n_len_bins)
    )
    plt.savefig(fn)


def process_cell_lines(cell_lines, extra_firsts, extra_lasts, ax=None):
    """ Processing after filtering """
    bins, n_len_bins = bin_cells(cell_lines, extra_firsts, extra_lasts)
    print("Number of cell length bins set to {0}".format(n_len_bins))

    bins_norm = {}
    for bin_num, cycle_bin in bins.items():
        norm = normalise_bin(cycle_bin, n_len_bins)
        bins_norm[bin_num] = norm

    plot_bins(bins_norm, n_len_bins, cycle_bins=bins, ax=ax)


if __name__ == "__main__":
    if os.path.exists("mt"):
        # we're in a processable directory already
        cell_lines = get_cell_lines()

    elif os.path.exists("wanted.json"):
        # we're not in a processable directory
        # but we can process all directories
        TLDs = json.loads(open("wanted.json").read()).keys()
        orig_dir = os.getcwd()
        cell_lines = []
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
                    cell_lines.extend(get_cell_lines())
                    os.chdir(orig_dir)
        print("*** Got {0} cell lines from all directories".format(len(cell_lines)))

    else:
        print("Nothing to do, exiting")
        sys.exit(1)

    cell_lines, (firsts, lasts) = filter_cell_lines(cell_lines)
    process_cell_lines(cell_lines, firsts, lasts)

