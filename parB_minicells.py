#!/usr/bin/env python

""" Determine whether cells without ParB foci are also minicells """

import glob
import json
import os
import re
import progressbar
import hashlib

import pandas as pd
import numpy as np

from analysis_lib import shared


DATASTRUCT = [
    "topdir", "subdir", "lineage_num",
    "cell_id", "parent_id", "daughter1_id", "daughter2_id",
    "num_init_spots", "num_final_spots",
    "length_birth", "length_division",
    "length_start", "length_end",
    "num_frames",
]
PX = 0.12254


def process_target(topdir, subdir):
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lineage_nums = [int(re.search("lineage(\d+).npy", x).group(1)) for x in lin_files]
    spot_data = []
    print("Processing {0}".format(os.getcwd()))
    progress = progressbar.ProgressBar()
    all_data = []
    for lineage_num, lf in progress(list(zip(lineage_nums, lin_files))):
        cell_line = np.load(lf)

        num_init_spots = len(cell_line[0].ParB)
        num_final_spots = len(cell_line[-1].ParB)

        len_start = cell_line[0].length[0][0]
        len_end = cell_line[-1].length[0][0]

        len_birth, len_division = None, None
        child1, child2 = None, None
        if cell_line[0].parent:
            len_birth = len_start
        if cell_line[-1].children:
            len_division = len_end
            child1, child2 = cell_line[-1].children

        data = pd.Series(
            [
                topdir, subdir, lineage_num,
                cell_line[0].id, cell_line[0].parent, child1, child2,
                num_init_spots, num_final_spots,
                len_birth, len_division,
                len_start, len_end,
                len(cell_line)
            ],
            index=DATASTRUCT,
            name=hashlib.sha1(
                "{0}{1}".format(os.getcwd(), lineage_num).encode("utf8")
            ).hexdigest()
        )
        all_data.append(data)
    return all_data


def main():
    # for all groupings, determine number of ParB spots (total) and at start
    # record num spots, num init spots, num final spots, cell length at birth
    # (or None), cell length at division (or None), cell length at start, cell
    # length at end
    data = pd.DataFrame(columns=DATASTRUCT)

    groups = json.loads(open("groupings.json").read())
    target_dirs = groups["delParAB"]
    orig_dir = os.getcwd()
    for target_dir in target_dirs:
        for subdir in os.listdir(target_dir):
            target = os.path.join(target_dir, subdir)
            target_conf = [
                os.path.join(target, "mt", "mt.mat"),
                os.path.join(target, "data", "cell_lines", "lineage01.npy"),
            ]
            conf = [os.path.exists(x) for x in target_conf]
            if os.path.isdir(target) and sum(conf) == len(conf):
                os.chdir(target)
                data = data.append(process_target(target_dir, subdir))
                os.chdir(orig_dir)

    print()
    print(data[data.num_init_spots == 0]["length_start"] * PX)


if __name__ == "__main__":
    main()
