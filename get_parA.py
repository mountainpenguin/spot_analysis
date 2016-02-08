#!/usr/bin/env python

import glob
import numpy as np
from spot_analysis import Analysis
import re

class Fake(object):
    def __init__(self):
        self.method = 1
        self.MEAN = 1
        self.SUM = 2
        self.MAX = 3


def main():
    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    for targetfile in targetfiles:
        cell_line = np.load(targetfile)
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        for cell in cell_line:
            f = cell.parB_img
            f_denoise = Analysis.denoise(Fake(), f)

            f2 = cell.parA_img
            f2_denoise = Analysis.denoise(Fake(), f2)

            data = f, f_denoise, f2, f2_denoise

            xs_l = cell.mesh[:, 0]
            ys_l = cell.mesh[:, 1]
            xs_r = cell.mesh[:, 2]
            ys_r = cell.mesh[:, 3]
            if not cell.orientation:
                xs_l = xs_l[::-1]
                ys_l = ys_l[::-1]
                xs_r = xs_r[::-1]
                ys_r = ys_r[::-1]

            mesh = xs_l, xs_r, ys_l, ys_r

            _, _, F2, F2_unsmoothed = Analysis.get_fluor_for_rib(
                Fake(), data, mesh
            )

            bg = f2.mean()
            m = (F2 - bg).mean() - (F2 - bg).std()

            cell.parA_fluorescence_smoothed = F2 - bg
            cell.parA_fluorescence_unsmoothed = F2_unsmoothed - bg

        np.save(
            "data/cell_lines/lineage{0:02d}".format(lineage_num),
            cell_line
        )
        print("Overwritten lineage{0:02d}".format(lineage_num))
    print("All done")


if __name__ == "__main__":
    main()
