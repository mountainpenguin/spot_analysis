#!/usr/bin/env python

import spot_plot
import spot_ancestry
import get_parA
import os
import json
import numpy as np
import sys


def ancestry():
    wanted = json.loads(open("wanted.json").read())
    original_path = os.path.abspath(os.getcwd())
    for d, v in wanted.items():
        for subdir in v:
            path = os.path.abspath(os.path.join(d, subdir))
            print("Processing path:", path)
            os.chdir(path)
            spot_ancestry.plot_lineages()
            os.chdir(original_path)


def normal():
    wanted = json.loads(open("wanted.json").read())
    original_path = os.path.abspath(os.getcwd())
    for d, v in wanted.items():
        for subdir, lins in v.items():
            path = os.path.abspath(os.path.join(d, subdir))
            print("Processing path:", path)
            os.chdir(path)

            for lin in lins:
                lineage_num = int(lin)
                target_file = os.path.join(
                    "data",
                    "cell_lines",
                    "lineage{0:02d}.npy".format(lineage_num)
                )
                cell_line = np.load(target_file)
                spot_plot.plot_images(cell_line, lineage_num)
                try:
                    spot_plot.plot_graphs(cell_line, lineage_num)
                except AttributeError:
                    print("ParA data missing")
                    get_parA.main()
                    cell_line = np.load(target_file)
                    spot_plot.plot_graphs(cell_line, lineage_num)

                print("Generated plots for lineage {0}".format(lineage_num))

            os.chdir(original_path)


if __name__ == "__main__":
    if "-ancestry" in sys.argv:
        ancestry()
    else:
        normal()
