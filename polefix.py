#!/usr/bin/env python

"""Fixes pole orientations of progenitor cells to have their new pole
    first"""

import glob
import numpy as np
import os

from lineage_lib import poles


def main():
    savedfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
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
            print(cell.id, end=" ")
            # convert orientation False
            # convert pole_assignment is -1

            if hasattr(cell, "orientation") and not cell.orientation:
                print("mesh reversal", end=" ")
                cell.mesh = cell.mesh[::-1]

            if pole_assignment == -1:
                print("mesh reversal2 and fluor reversal")
                cell.mesh = cell.mesh[::-1]
                # conversions apply to:
                #   M_x, M_y
                #   ParA (x, _, _)
                #   ParB (x, _, _)
                #   mesh
                #   parA_fluorescence_smoothed
                #   parA_fluorescence_unsmoothed
                #   parB_fluorescence_smoothed
                #   parB_fluorescence_unsmoothed

#                cell.M_x = cell.M_x[::-1]
#                cell.M_y = cell.M_y[::-1]

                cell.ParA = (cell.length[0][0] - cell.ParA[0], cell.ParA[1], cell.ParA[2])
                cell.parA_fluorescence_smoothed = cell.parA_fluorescence_smoothed[::-1]
                cell.parA_fluorescence_unsmoothed = cell.parA_fluorescence_unsmoothed[::-1]

                ParB = []
                for pb in cell.ParB:
                    ParB.append((
                        cell.length[0][0] - pb[0],
                        pb[1],
                        pb[2]
                    ))
                cell.ParB = ParB
                cell.parB_fluorescence_smoothed = cell.parB_fluorescence_smoothed[::-1]
                cell.parB_fluorescence_unsmoothed = cell.parB_fluorescence_unsmoothed[::-1]
        print()

        # save modified data
        np.save(
            sf, cell_line
        )

        # regenerate spot_plots
        # regenerate spot_ancestry plots


if __name__ == "__main__":
    main()
