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

n_bins = 7

CHECK n_bins ASSIGNMENT IN `bin_cells`


def get_cell_lines():
    # for now, just deal with one dir at a time
    lin_files = glob.glob("data/cell_lines/lineage*.npy")
    cell_lines = []
    for lf in lin_files:
        cell_line = np.load(lf)
        cell_lines.append(cell_line)
    print("Got {0} cell lines".format(len(cell_lines)))
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
        for cell in cell_line:
            segments = len(cell.mesh)
            if segments > n_len_bins:
                n_len_bins = int(segments)
            if bin_count >= per_bin:
                bin_num += 1
                bin_count = 0
            if bin_num >= n_bins - 1:
                bin_num -= 1

            bins[bin_num].append(cell)
            bin_count += 1

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
        if normalised is None:
            normalised = rescaled_vals
        else:
            normalised += rescaled_vals
    normalised = normalised / normalised.max()
    return normalised


def plot_bins(bins, n_len_bins, ax=None):
    cmapper = plt.cm.get_cmap("afmhot")
    vals = np.array([x for x in bins.values()])
    max_inten = vals.max()
    if not ax:
        plt.figure()
        ax = plt.subplot(111)
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
            ax.add_patch(r)
            i += 1
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])


def process_cell_lines(cell_lines, extra_firsts, extra_lasts, ax=None):
    """ Processing after filtering """
    bins, n_len_bins = bin_cells(cell_lines, extra_firsts, extra_lasts)
    print("Number of cell length bins set to {0}".format(n_len_bins))

    bins_norm = {}
    for bin_num, cycle_bin in bins.items():
        norm = normalise_bin(cycle_bin, n_len_bins)
        bins_norm[bin_num] = norm

    plot_bins(bins_norm, n_len_bins, ax=ax)
    return n_len_bins


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
    n_len_bins = process_cell_lines(cell_lines, firsts, lasts)
    plt.xlabel("% of cell cycle")
    plt.ylabel("% distance from new pole")
    sns.despine()
    plt.savefig("C={0},L={1},ParA_distribution.pdf".format(n_bins, n_len_bins))
