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
n_len_bins = 31


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


    return bins


def normalise_bin(cells):
    """ normalise ParA by cell length """
    # split cell in 30 segments
    # return ParA values as % of cell length
    # normalise by intensity
    normalised = None
    for cell in cells:
        pos_bins = range(n_len_bins)
        max_pos = len(cell.parA_fluorescence_unsmoothed)
        scaling = max_pos / n_len_bins
        vals = np.array(range(len(cell.parA_fluorescence_unsmoothed))) / scaling
        freq, _ = np.histogram(vals, bins=pos_bins, weights=cell.parA_fluorescence_unsmoothed)
        if normalised is None:
            normalised = freq
        else:
            normalised += freq
    normalised = normalised / normalised.max()
    return normalised


def plot_bins(bins, ax=None):
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
    bins = bin_cells(cell_lines, extra_firsts, extra_lasts)

    bins_norm = {}
    for bin_num, cycle_bin in bins.items():
        norm = normalise_bin(cycle_bin)
        bins_norm[bin_num] = norm

    plot_bins(bins_norm, ax)


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
    plt.xlabel("% of cell cycle")
    plt.ylabel("% distance from new pole")
    sns.despine()
    plt.savefig("ParA_distribution.pdf")
