#!/usr/bin/env python

""" Script to plot ParB distribution at birth

Bins cells according to how many ParB foci they are born with
Plots ParB position as percentage of cell length for each bin
"""

import glob
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc as mpl_rc

sns.set_style("white")
sns.set_context("paper")
mpl_rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl_rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]

PX = 0.12254

def process(data_columns):
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lineage_nums = sorted([int(re.search("lineage(\d+).npy", x).group(1)) for x in lin_files])

    output_data = pd.DataFrame(columns=data_columns)
    num_cells = 0
    for lineage_num in lineage_nums:
        print("lineage {0} of {1}".format(lineage_num, len(lineage_nums)), end=", ")
        lineage_file = "data/cell_lines/lineage{0:02d}.npy".format(lineage_num)
        cell_line = np.load(lineage_file)
#        T = cell_line[0].T
        print(
            "({0} ParB spots and poles {1})".format(
                len(cell_line[0].ParB),
                cell_line[0].pole_assignment
            ), end=", "
        )
        if cell_line[0].pole_assignment and cell_line[0].ParB:
            num_cells += 1
            for d_newpole, intensity, cell_length in cell_line[0].ParB:
                # distance from new pole, intensity, cell_length
                # print(sorted(cell_line[0].__dict__.keys()))
                d_midcell = (cell_length / 2) - d_newpole

#                if d_midcell > 0:
#                    newpole_associated = True
#                    oldpole_associated = False
#                elif d_midcell < 0:
#                    newpole_associated = False
#                    oldpole_associated = True
#                else:
#                    newpole_associated = False
#                    oldpole_associated = False

                position_perc = 100 * d_newpole / cell_length
                output_data = output_data.append(
                    pd.Series(dict(zip(
                        data_columns,
                        [
                            lineage_num,
                            cell_length * PX,
                            d_newpole * PX,
                            d_midcell * PX,
                            np.abs(d_midcell) * PX,
                            position_perc,
                            len(cell_line[0].ParB),
                        ]
                    ))),
                    ignore_index=True
                )
        print("got {0} cells and {1} spots".format(num_cells, len(output_data)))
    return output_data


if __name__ == "__main__":
    data_columns = [
        "lineage",
        "cell_length",
        "d_newpole",
        "d_midcell",
        "abs_d_midcell",
        "position_perc",
        "num_spots",
    ]

    groups = json.loads(open("groupings.json").read())
    orig_dir = os.getcwd()
    if os.path.exists("parB_distribution_results/data.pandas") and "-f" not in sys.argv:
        data = pd.read_pickle("parB_distribution_results/data.pandas")
    else:
        data = pd.DataFrame(columns=data_columns)
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
                        print(" Handling {0}".format(target))
                        result = process(data_columns)
                        result["source"] = [target] * len(result)
                        result["prefix"] = [prefix] * len(result)
                        data = data.append(result, ignore_index=True)
                        os.chdir(orig_dir)

        if not os.path.exists("parB_distribution_results"):
            os.mkdir("parB_distribution_results")
        data.to_pickle("parB_distribution_results/data.pandas")

    fig = plt.figure(figsize=(6, 20))
    max_spot_count = data.num_spots.max()
    ax = None
    for spot_count in data.num_spots.unique():
        if ax:
            ax = fig.add_subplot(max_spot_count, 1, spot_count, sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(max_spot_count, 1, spot_count)
        for dataset in data.prefix.unique():
            spots = data[(data.num_spots == spot_count) & (data.prefix == dataset)]
            ax.plot(spots.position_perc, spots.cell_length, ls="none", marker="o", label=dataset)
        ax.set_xlim([0, 100])
        ax.set_ylabel(r"\textbf{{{0} ParB foci}}\\Cell Length (\si{{\micro\metre}})".format(int(spot_count)))
        ax.set_xlabel(r"Percentage distance from new pole")
        if spot_count == 1:
            ax.legend()
        sns.despine()

    fig.tight_layout()
    plt.savefig("parB_distribution_results/ParB_distribution.pdf")

