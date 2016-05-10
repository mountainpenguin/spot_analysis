#!/usr/bin/env python

"""Fixes pole orientations of progenitor cells to have their new pole
    first"""

import glob
import numpy as np
import os
import shutil
import datetime
import json

from lineage_lib import poles


def main():
    # backup data
    print("Backing up data directory")
    shutil.copytree(
        "data",
        datetime.datetime.strftime(
            datetime.datetime.now(), "data.backup.%d.%m.%y-%H%M"
        )
    )

    savedfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    print("Assigning poles")
    P = poles.PoleAssign()
    poles_ref = P.assign_poles()
    for sf in savedfiles:
        print("sf:", sf)
        cell_line = np.load(sf)
        spot_data_file = os.path.join(
            "data",
            "spot_data",
            os.path.basename(sf)
        )
        if os.path.exists(spot_data_file):
            # remove it
            os.remove(spot_data_file)

        pole_assignment = poles_ref[cell_line[0].id]
        for cell in cell_line:
            print(cell.id)
            # convert orientation False
            # convert pole_assignment is -1

            if pole_assignment == -1:
                reversal(cell, pole_assignment)

            # delete orientation parameter
            # add pole_assignment parameter
            delattr(cell, "orientation")
            cell.pole_assignment = 0

        # adjust poles.json
        poles_ref[cell_line[0].id] = 0
        open("poles.json", "w").write(json.dumps(poles_ref))

        # save modified data
        np.save(
            sf, cell_line
        )

        # regenerate spot_plots
        # regenerate spot_ancestry plots


def reversal(cell, pole):
    # reverse everything
    cell.M_x = cell.M_x[::-1]
    cell.M_y = cell.M_y[::-1]
    cell.mesh = cell.mesh[::-1]
#    if pole == -1:
#        reversal(cell, 0)
    cell.parA_fluorescence_smoothed = cell.parA_fluorescence_smoothed[::-1]
    cell.parA_fluorescence_unsmoothed = cell.parA_fluorescence_unsmoothed[::-1]
    cell.parB_fluorescence_smoothed = cell.parB_fluorescence_smoothed[::-1]
    cell.parB_fluorescence_unsmoothed = cell.parB_fluorescence_unsmoothed[::-1]
    cell.ParA = (cell.length[0][0] - cell.ParA[0], cell.ParA[1], cell.ParA[2])
    ParB = []
    for pb in cell.ParB:
        ParB.append((cell.length[0][0] - pb[0], pb[1], pb[2]))
    cell.ParB = ParB


if __name__ == "__main__":
    main()
