#!/usr/bin/env python


import xlwt
from analysis_lib import shared
import glob
import re
import numpy as np
import os
import datetime


def gen_xl(cell_line, lineage_num):
    lineage_num = int(lineage_num)
    T = cell_line[0].T
    t = cell_line[0].t

    spots_ParA = [x.ParA for x in cell_line]
    spots_ParB = shared.get_parB_path(cell_line, T)

    excel_wb = xlwt.Workbook()
    ws_parA = excel_wb.add_sheet("ParA")
    ws_parB = excel_wb.add_sheet("ParB")

    ws_parA.write(0, 0, "Time")
    ws_parA.write(0, 1, "Distance from top pole")
    ws_parA.write(0, 2, "Distance from midcell")
    ws_parA.write(0, 3, "Cell length")
    ws_parA.write(0, 4, "Spot intensity")

    parAs = {}
    i = 0
    for pos, inten, celllen in spots_ParA:
        midcell = celllen / 2
        spot = pos - midcell

        ws_parA.write(i + 1, 0, int(t[i]))
        ws_parA.write(i + 1, 1, float(pos))
        ws_parA.write(i + 1, 2, float(spot))
        ws_parA.write(i + 1, 3, float(celllen))
        ws_parA.write(i + 1, 4, float(inten))
        parAs[t[i]] = spot

        ws_parB.write(i + 1, 1, float(celllen))

        i += 1

    ws_parB.write(0, 0, "Time")
    ws_parB.write(0, 1, "Cell length")

    i = 1
    time_dict = {}
    for time_ in t:
        ws_parB.write(i, 0, int(time_))
        time_dict[float(time_)] = i
        i += 1

    rmax = max(time_dict.values())
    ws_parB.write(rmax + 2, 0, "Intensity mean:")
    ws_parB.write(rmax + 3, 0, "Intensity SEM:")
    ws_parB.write(rmax + 4, 0, "Distance from ParA (mean):")
    ws_parB.write(rmax + 5, 0, "Distance from ParA (SEM):")

    col = 2
    spotnum = 1
    for x in spots_ParB:
        s = x.spots(False)
        ws_parB.write(0, col, "Distance from mid-cell (Spot {0})".format(spotnum))
        ws_parB.write(0, col + 1, "Intensity (Spot {0})".format(spotnum))
        ws_parB.write(0, col + 2, "Distance from ParA (Spot {0})".format(spotnum))

        parA_d = []
        for spot in s:
            r = time_dict[spot[0]]
            ws_parB.write(r, col, float(spot[1]))
            ws_parB.write(r, col + 1, float(spot[2]))
            dpA = spot[1] - parAs[spot[0]]
            parA_d.append(dpA)
            ws_parB.write(r, col + 2, float(dpA))

        # intensity mean and SEM for spot lineage
        intensity_mean = s[:, 2].mean()
        intensity_sem = s[:, 2].std() / np.sqrt(len(s[:, 1]))
        ws_parB.write(rmax + 2, col + 1, float(intensity_mean))
        ws_parB.write(rmax + 3, col + 1, float(intensity_sem))

        # distance from ParA mean and SEM for spot lineage
        parA_dmean = np.mean(parA_d)
        parA_dsem = np.std(parA_d) / np.sqrt(len(parA_d))
        ws_parB.write(rmax + 4, col + 2, float(parA_dmean))
        ws_parB.write(rmax + 5, col + 2, float(parA_dsem))

        col += 3
        spotnum += 1

    if not os.path.exists("data/xls"):
        os.mkdir("data/xls")

    fn = "data/xls/lineage{0:02d}.xls".format(lineage_num)

    if os.path.exists(fn):
        if not os.path.exists("data/xls/backups"):
            os.mkdir("data/xls/backups")

        backup_file = "data/xls/backups/{0}-lineage{1:02d}.xls".format(
            datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M"),
            lineage_num
        )
        attempt = 2
        while os.path.exists(backup_file):
            first = backup_file.split(".xls")[0]
            backup_file = "{0}.xls-{1}".format(first, attempt)
            attempt += 1
        print("Backing up previous xls file to {0}".format(backup_file))
        os.rename(fn, backup_file)

    print("Saving new xls file to {0}".format(fn))
    excel_wb.save(fn)


def main():
    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    for targetfile in targetfiles:
        cell_line = np.load(targetfile)
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        gen_xl(cell_line, lineage_num)

if __name__ == "__main__":
    main()
