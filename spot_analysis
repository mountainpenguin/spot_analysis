#!/usr/bin/env python

from lineage_lib import track
from analysis_lib import colourmap
from analysis_lib import shared
import spot_plot
import sys
import glob
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import scipy.signal
import scipy.stats
import numpy as np
import operator
import seaborn as sns
import os
import random
import string
import hashlib
import xlwt
import peakutils

# PX_UM = 0.12254  # 1px in um for 63x objective on WF2


class SpotTimeLapse(object):
    def __init__(self, t, p, i, l):
        self.id = hashlib.sha1("".join([
            random.choice(
                string.ascii_letters + string.digits
            ) for x in range(40)
        ]).encode("utf8")).hexdigest()
        self.timing = [t]
        if p < 0:
            # bottom pole
            self.POLE = -1
        else:
            self.POLE = 1
        self.positions = [p]
        self.intensity = [i]
        self.lengths = [l]

    def spots(self, adjust=True):
        if adjust:
            return np.array([
                self.timing,
                self.POLE * np.array(self.positions),
                self.intensity
            ]).T
        else:
            return np.array([
                self.timing,
                self.positions,
                self.intensity
            ]).T

    def len(self):
        return np.array(self.lengths)

    def last(self):
        return np.array([
            self.timing[-1], self.positions[-1], self.intensity[-1]
        ])

    def append(self, t, p, i, l):
        self.timing.append(t)
        self.positions.append(p)
        self.intensity.append(i)
        self.lengths.append(l)

    def __iadd__(self, other):
        self.timing.append(other[0])
        self.positions.append(other[1])
        self.intensity.append(other[2])
        self.lengths.append(other[3])

    def __len__(self):
        return len(self.lengths)


def deaxis(ax=None):
    if not ax:
        ax = plt.gca()
    ax.axis("off")


def despine(ax=None, keep=0):
    if not ax:
        ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    if keep == 1:
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
    elif keep == 0:
        ax.spines["left"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")


class Analysis(track.Lineage):
    MEAN = 1
    SUM = 2
    MAX = 3

    def __init__(self, debug, method=MEAN, skip=True, noimage=False):
        self.debug = debug
        self.method = method
        self.SKIP = skip
        self.NO_IMAGE = noimage
        track.Lineage.__init__(self)

    def apply_alignments(self):
        f = 0
        for align in self.alignment:
            try:
                cells = self.frames[f].cells
            except IndexError:
                break
            for c in cells:
                c.mesh = c.mesh - [
                    align[1], align[0], align[1], align[0]
                ]
            f += 1

    def process_fluor(self, img):
        # Gaussian smoothing to denoise
        img = self.denoise(img)
        return img

    def denoise(self, img):
        out = scipy.ndimage.gaussian_filter(img, 2)
        return out

    def get_fluor_for_rib(self, data, mesh):
        f, f_denoise, f2, f2_denoise = data
        xs_l, xs_r, ys_l, ys_r = mesh
        i = 0
        F = []  # average fluor along each mesh 'rib'
        F_unsmoothed = []

        F2 = []
        F2_unsmoothed = []

        while i < xs_l.shape[0]:
            H = int(np.sqrt(
                (ys_r[i] - ys_l[i]) ** 2 +
                (xs_r[i] - xs_l[i]) ** 2
            )) + 1
            x = np.linspace(
                xs_l[i], xs_r[i], H
            )
            y = np.linspace(
                ys_l[i], ys_r[i], H
            )
            z = scipy.ndimage.map_coordinates(
                f_denoise, np.vstack((y, x))
            )
            z_unsmoothed = scipy.ndimage.map_coordinates(
                f, np.vstack((y, x))
            )
            z2 = scipy.ndimage.map_coordinates(
                f2_denoise, np.vstack((y, x))
            )
            z2_unsmoothed = scipy.ndimage.map_coordinates(
                f2, np.vstack((y, x))
            )

            if self.method == self.MEAN:
                F.append(z.mean())
                F_unsmoothed.append(z_unsmoothed.mean())
                F2.append(z2.mean())
                F2_unsmoothed.append(z2_unsmoothed.mean())
            elif self.method == self.SUM:
                F.append(z.sum())
                F_unsmoothed.append(z_unsmoothed.sum())
                F2.append(z2.sum())
                F2_unsmoothed.append(z2_unsmoothed.sum())
            elif self.method == self.MAX:
                F.append(z.max())
                F_unsmoothed.append(z_unsmoothed.max())
                F2.append(z2.max())
                F2_unsmoothed.append(z2_unsmoothed.max())
            i += 1
        return F, F_unsmoothed, F2, F2_unsmoothed

    def peak_dist(self, x, l):
        if x < l:
            return x
        else:
            return l - x

        # _p = lambda x: if x > len(F) // 2: return len(F) - x else return x

    def get_spots(self, cell, orientation=True):
        frame_idx = cell.frame - 1
        xs_l = cell.mesh[:, 0]
        ys_l = cell.mesh[:, 1]
        xs_r = cell.mesh[:, 2]
        ys_r = cell.mesh[:, 3]

        if not orientation:
            xs_l = xs_l[::-1]
            ys_l = ys_l[::-1]
            xs_r = xs_r[::-1]
            ys_r = ys_r[::-1]

        M_x = (xs_l + xs_r) / 2
        M_y = (ys_l + ys_r) / 2

        p = scipy.misc.imread(self.phase[frame_idx])
        f = scipy.misc.imread(self.fluor[frame_idx])
        f_denoise = self.process_fluor(f)

#        plt.figure()
#
#        ax = plt.subplot(131)
#        plt.imshow(p, cmap=colourmap.cm)
#        plt.plot(xs_l, ys_l, "w-")
#        plt.plot(xs_r, ys_r, "w-")
#
#        plt.subplot(132, sharex=ax, sharey=ax)
#        plt.imshow(f, cmap=colourmap.cm)
#        plt.plot(xs_l, ys_l, "w-")
#        plt.plot(xs_r, ys_r, "w-")
#
#        plt.subplot(133, sharex=ax, sharey=ax)
#        plt.imshow(f_denoise, cmap=colourmap.cm)
#        plt.plot(xs_l, ys_l, "w-")
#        plt.plot(xs_r, ys_r, "w-")
#
#        plt.show()

        f2 = scipy.misc.imread(self.fluor2[frame_idx])  # f2 = ParA
        f2_denoise = self.process_fluor(f2)

        data = f, f_denoise, f2, f2_denoise
        mesh = xs_l, xs_r, ys_l, ys_r

        F, F_unsmoothed, F2, F2_unsmoothed = self.get_fluor_for_rib(
            data, mesh
        )

        # M = np.column_stack((M_x, M_y))
        i = np.array(range(len(F)))
        bg = f.mean()
        bg2 = f2.mean()

        # get peaks
        m = (F - bg).mean() - np.std(F - bg)
        m2 = (F2 - bg2).mean() - np.std(F2 - bg2)
        # s = np.std(F)

        cell.parB_fluorescence_smoothed = F - bg
        cell.parB_fluorescence_unsmoothed = F_unsmoothed - bg

        # argrelextrema peak-finding method
#        np.save("test{0:02d}".format(cell.frame - 19), [F - bg, F_unsmoothed - bg])
#        peaks = scipy.signal.argrelextrema(F - bg, np.greater_equal)[0]

        # peakutil method
        peaks1 = peakutils.indexes(F - bg, thres=0.3, min_dist=5)
        model = 20
        noisy_peaks = [
            (F - bg) + np.random.normal(0, (F - bg).std() / 2, (F - bg).shape)
            for x in range(20)
        ]
        smoothed_noisy = [
            scipy.ndimage.gaussian_filter(x, 2)
            for x in noisy_peaks
        ]
        noisy_indexes = [
            peakutils.indexes(x, thres=0.3, min_dist=5)
            for x in smoothed_noisy
        ]
        selected_indexes = {}
        for noisy_indices in noisy_indexes + [peaks1]:
            for noisy_index in noisy_indices:
                if noisy_index in selected_indexes:
                    selected_indexes[noisy_index] += 1
                else:
                    selected_indexes[noisy_index] = 1
        noisy_bins = {}
        for bin_low in range(selected_indexes and max(selected_indexes.keys()) or 0):
            binrange = range(bin_low, bin_low + 3)
            for noisy_pos, noisy_count in selected_indexes.items():
                if noisy_pos in binrange and binrange in noisy_bins:
                    noisy_bins[binrange] += noisy_count
                elif noisy_pos in binrange:
                    noisy_bins[binrange] = noisy_count

        filtered_indexes = [
            x[0] for x in noisy_bins.items() if x[1] >= model / 2
        ]
        peaks = []
        for index in peaks1:
            for index_range in filtered_indexes:
                if index in index_range and index not in peaks:
                    peaks.append(index)

        ParB_vals = [
            (i[_z], (F - bg)[_z]) for _z in peaks if (F - bg)[_z] > m
        ]

        # ParA peak is the maximum value
        _parA = np.array(F2).argmax()
        ParA_max = _parA, F2[_parA]

        # check distances between peaks
        if len(ParB_vals) > 1:
            # group peaks into multiples of 10 for x position
            # i.e. 0-9, 10-19, etc. are grouped together
            modval = 5
            modded = [(_i[0] // modval, (_i[0], _i[1])) for _i in ParB_vals]

            # rearrange flat list of modded groups into grouped list
            grouped = set(map(lambda x: x[0], modded))
            groups = []
            for group in grouped:
                out = []
                for mod_val in modded:
                    if mod_val[0] == group:
                        out.append(mod_val[1])
                groups.append(out)

            out = []
            for group in groups:
                if len(group) > 1:
#                    # average
#                    index = int(np.mean([_val[0] for _val in group]))
#                    out.append((index, group[0][1]))
                    # use highest intensity value of group
                    out.append(max(group, key=operator.itemgetter(1)))
                else:
                    out.append(group[0])
            ParB_vals = sorted(out, key=operator.itemgetter(0))

            # remove any spot that is within 2 pixels of a pole

            ParB_vals = [
                _i for _i in ParB_vals
                if self._pole_dist_check(_i, i, cell.length[0][0])
            ]

            # if more than 2 spots, select two with greatest intensity
#            if len(ParB_vals) > 2:
#                ParB_vals = sorted([
#                    ParB_vals.pop(ParB_vals.index(max(ParB_vals, key=operator.itemgetter(1)))),
#                    ParB_vals.pop(ParB_vals.index(max(ParB_vals, key=operator.itemgetter(1))))
#                ], key=operator.itemgetter(0))

#        # calculate ParA signal (f2)
#        # split cell into 8 segments
#        # determine total intensity of fluorescence in that region
#        num_ribs = len(cell.mesh)
#        interval = num_ribs / 8
#        seg_bounds = np.arange(
#            interval,
#            num_ribs + interval,
#            interval,
#            dtype=np.int
#        )
#        seg_polygons = []
#        for seg_bound in seg_bounds:
#            seg_mesh = cell.mesh[seg_bound - interval:seg_bound]
#            seg_xs_l = seg_mesh[:, 0]
#            seg_xs_r = seg_mesh[:, 2]
#            seg_ys_l = seg_mesh[:, 1]
#            seg_ys_r = seg_mesh[:, 3]
#            seg_mesh = seg_xs_l, seg_xs_r, seg_ys_l, seg_ys_r
#            _1, _2, seg_F2, _3 = self.get_fluor_for_rib(
#                data, seg_mesh
#            )
#            seg_total_F2 = np.sum(seg_F2)
#            seg_polygons.append(seg_total_F2)

        ParA_val = ((ParA_max[0] / i[-1]) * cell.length[0][0], ParA_max[1], cell.length[0][0])
        ParB_vals = [
            ((__[0] / i[-1]) * cell.length[0][0], __[1], cell.length[0][0]) for __ in ParB_vals
        ]

        if self.debug:
            plt.figure()
            ax = plt.subplot(231)
            plt.imshow(p, cmap=plt.cm.gray)
            # plt.plot(M_x, M_y, "r-")
            plt.plot(xs_l, ys_l, "w-", lw=0.5)
            plt.plot(xs_r, ys_r, "w-", lw=0.5)
#            _ = 0
#            while _ < xs_l.shape[0]:
#                plt.plot(
#                    (xs_l[_], xs_r[_]),
#                    (ys_l[_], ys_r[_]),
#                    "y-"
#                )
#                _ += 1
#
            for p in ParB_vals:
                x, y = M_x[p[0]], M_y[p[0]]
                plt.plot(
                    x, y, "r*"
                )
                x, y = M_x[ParA_max[0]], M_y[ParA_max[0]]
                plt.plot(
                    x, y, "y*"
                )

            plt.title("Phase")
            deaxis()

            plt.subplot(232, sharex=ax, sharey=ax)
            plt.imshow(f2, cmap=colourmap.cm)
            plt.plot(xs_l, ys_l, "w-", lw=0.5)
            plt.plot(xs_r, ys_r, "w-", lw=0.5)
            x, y = M_x[ParA_max[0]], M_y[ParA_max[0]]
            plt.plot(
                x, y, "w*"
            )
            plt.title("ParA")
            deaxis()

            plt.subplot(233, sharex=ax, sharey=ax)
            plt.imshow(f, cmap=colourmap.cm)
            plt.plot(xs_l, ys_l, "w-", lw=0.5)
            plt.plot(xs_r, ys_r, "w-", lw=0.5)

            for p in ParB_vals:
                x, y = M_x[p[0]], M_y[p[0]]
                plt.plot(
                    x, y, "w*"
                )
            plt.title("ParB")
            plt.xlim([0, f.shape[0]])
            plt.ylim([0, f.shape[1]])
            deaxis()

            plt.subplot(235)
            plt.plot(i, F2 - bg2, "k-")
            plt.plot(i, F2_unsmoothed - bg2, "k-", alpha=0.4)
            plt.plot(
                ParA_max[0], ParA_max[1] - bg2, "r."
            )
            plt.plot([i[0], i[-1]], [m2, m2])

            plt.subplot(236)
            plt.plot(i, F - bg, "k-")
            plt.plot(i, F_unsmoothed - bg, "k-", alpha=0.4)
            plt.plot([i[0], i[-1]], [m, m])
            # plt.plot([i[0], i[-1]], [m + s, m + s], "y-")

            for p in ParB_vals:
                plt.plot(
                    p[0], p[1], "r."
                )

            plt.show()

        return ParA_val, ParB_vals

    def _pole_dist_check(self, pos, arr, clen, threshold=2):
        p1_d = (pos[0] / arr[-1]) * clen
        p2_d = clen - p1_d
        return p1_d > threshold and p2_d > threshold

    def get_orientation(self, cell, prior):
        poles = cell.mesh[0, 0:2], cell.mesh[-1, 0:2]
        d1 = np.sqrt(
            ((poles[0][0] - prior[0][0]) ** 2) +
            ((poles[0][1] - prior[0][1]) ** 2)
        ) + np.sqrt(
            ((poles[1][0] - prior[1][0]) ** 2) +
            ((poles[1][1] - prior[1][1]) ** 2)
        )

        d2 = np.sqrt(
            ((poles[0][0] - prior[1][0]) ** 2) +
            ((poles[0][1] - prior[1][1]) ** 2)
        ) + np.sqrt(
            ((poles[1][0] - prior[0][0]) ** 2) +
            ((poles[1][1] - prior[0][1]) ** 2)
        )
        if d1 < d2:
            return True
        else:
            return False

    def _crop(self, im, xmin, xmax, ymin, ymax):
        return im[
            xmin:xmax,
            ymin:ymax
        ]

    def spot_finder(self, phase, fluor, fluor2, cell_lines):
        self.phase = phase
        self.fluor = fluor
        self.fluor2 = fluor2
        self.T = shared.get_timings()
        cell_line_num = 1
        for cell_line in cell_lines:
            if self.SKIP and os.path.exists("data/data-lineage{0}.pdf".format(cell_line_num)):
                print("Skipping cell lineage {0} of {1}".format(
                    cell_line_num, len(cell_lines)
                ))
                cell_line_num += 1
                continue

            L = []
            S_A = []
            S_B = []
            prior = None
            for cell in cell_line:
                L.append(cell.length[0][0])
                if prior:
                    orientation = self.get_orientation(cell, prior)
                else:
                    orientation = True
                cell.orientation = orientation
                spot_ParA, spots_ParB = self.get_spots(cell, orientation)
                cell.ParA = spot_ParA
                # spot_ParA:
                #  distance from pole
                #  intensity
                #  cell length

                cell.ParB = spots_ParB
                # each spot_ParB:
                #  distance from pole
                #  intensity
                #  cell length
                S_A.append(spot_ParA)
                S_B.append(spots_ParB)
                if orientation:
                    prior = cell.mesh[0, 0:2], cell.mesh[-1, 0:2]
                else:
                    prior = cell.mesh[-1, 0:2], cell.mesh[0, 0:2]

            start = cell_line[0].frame - 1
            end = cell_line[-1].frame - 1
            t = np.array(self.T[start:end + 1])

            # show cells
            frame_num = start + 1 - 1
            sp_num = 0
            while frame_num <= end:
                mesh = cell_line[sp_num].mesh
                if cell_line[sp_num].orientation:
                    M_x = (mesh[:, 0] + mesh[:, 2]) / 2
                    M_y = (mesh[:, 1] + mesh[:, 3]) / 2
                else:
                    M_x = (mesh[:, 0][::-1] + mesh[:, 2][::-1]) / 2
                    M_y = (mesh[:, 1][::-1] + mesh[:, 3][::-1]) / 2

                cell_line[sp_num].M_x = M_x
                cell_line[sp_num].M_y = M_y

                phase = scipy.misc.imread(self.phase[frame_num])
                parB = scipy.misc.imread(self.fluor[frame_num])
                parA = scipy.misc.imread(self.fluor2[frame_num])
                cell_line[sp_num].phase_img = phase
                cell_line[sp_num].parB_img = parB
                cell_line[sp_num].parA_img = parA

                bg_parA = parA - parA.mean()
                bg_parA[bg_parA < 0] = np.NaN
                cell_line[sp_num].parA_img_bg = bg_parA

                bg_parB = parB - parB.mean()
                bg_parB[bg_parB < 0] = np.NaN
                cell_line[sp_num].parB_img_bg = bg_parB

                frame_num += 1
                sp_num += 1

            if not os.path.exists("data"):
                os.mkdir("data")

            if not os.path.exists("data/cell_lines"):
                os.mkdir("data/cell_lines")

            np.save(
                "data/cell_lines/lineage{0:02d}".format(cell_line_num),
                cell_line
            )

            # plot cell lengths with cell-centre at 0
            L = np.array(L)

            excel_wb = xlwt.Workbook()
            ws_parA = excel_wb.add_sheet("ParA")
            ws_parB = excel_wb.add_sheet("ParB")

            ws_parA.write(0, 0, "Time")
            ws_parA.write(0, 1, "Distance from top pole")
            ws_parA.write(0, 2, "Distance from midcell")
            ws_parA.write(0, 3, "Cell length")
            ws_parB.write(0, 1, "Cell length")
            ws_parA.write(0, 4, "Spot intensity")

            parAs = {}
            i = 0
            for s in S_A:
                midcell = s[2] / 2
                spot = s[0] - midcell
                ws_parA.write(i + 1, 0, int(t[i]))
                ws_parA.write(i + 1, 1, float(s[0]))
                ws_parA.write(i + 1, 2, float(spot))
                ws_parA.write(i + 1, 3, float(s[2]))
                ws_parB.write(i + 1, 1, float(s[2]))
                ws_parA.write(i + 1, 4, float(s[1]))
                parAs[t[i]] = spot
                i += 1

            spots_ParB = shared.get_parB_path(cell_line, self.T)

            ws_parB.write(0, 0, "Time")
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

                dparA = []
                for spot in s:
                    r = time_dict[spot[0]]
                    ws_parB.write(r, col, float(spot[1]))
                    ws_parB.write(r, col + 1, float(spot[2]))
                    dpA = spot[1] - parAs[spot[0]]
                    dparA.append(dpA)
                    ws_parB.write(r, col + 2, float(dpA))

                # intensity mean and SEM for spot lineage
                x.intensity_mean = s[:, 2].mean()
                x.intensity_sem = s[:, 2].std() / np.sqrt(len(s[:, 1]))
                ws_parB.write(rmax + 2, col + 1, float(x.intensity_mean))
                ws_parB.write(rmax + 3, col + 1, float(x.intensity_sem))

                # distance from ParA mean and SEM for spot lineage
                x.parA_d = dparA
                x.parA_dmean = np.mean(dparA)
                x.parA_dsem = np.std(dparA) / np.sqrt(len(dparA))
                x.spotnum = spotnum
                ws_parB.write(rmax + 4, col + 2, float(x.parA_dmean))
                ws_parB.write(rmax + 5, col + 2, float(x.parA_dsem))

                col += 3
                spotnum += 1

            print("Analysed lineage {0}".format(cell_line_num))
            if not self.NO_IMAGE:
                spot_plot.plot_images(cell_line, cell_line_num)
                print("Generated images for lineage {0}".format(cell_line_num))
                spot_plot.plot_graphs(cell_line, cell_line_num)
                print("Generated graphs for lineage {0}".format(cell_line_num))
            excel_wb.save("data/lineage{0:02d}.xls".format(cell_line_num))
            print("Processed cell lineage {0} of {1}".format(
                cell_line_num, len(cell_lines)
            ))
            cell_line_num += 1


if __name__ == "__main__":
    sns.set_style("white")
    sns.set_context("talk")
    if "-d" in sys.argv:
        debug = True
    else:
        debug = False

    if "-s" in sys.argv:
        skip = False
    else:
        skip = True

    if "-n" in sys.argv:
        noimage = True
    else:
        noimage = False

    A = Analysis(debug=debug, skip=skip, noimage=noimage)
    A.apply_alignments()

    cell_lines = []
    initial_cells = A.frames[0].cells
    for f in initial_cells:
        lin = [f]
        while True:
            if type(f.children) is list:
                initial_cells.append(A.frames.cell(f.children[0]))
                initial_cells.append(A.frames.cell(f.children[1]))
                break
            elif f.children:
                c = f.children
                c = A.frames.cell(c)
                lin.append(c)
                f = c
            else:
                break
        cell_lines.append(lin)

#    final_cells = A.frames[-1].cells
#    cell_lines = []
#    for f in final_cells:
#        lin = [f]
#        while True:
#            if not f.parent:
#                break
#            p = f.parent
#            p = A.frames.cell(p)
#            lin.insert(0, p)
#            f = p
#        cell_lines.append(lin)

    phase = sorted(glob.glob("B/*.tif"))
    parA = sorted(glob.glob("F1/*.tif"))
    parB = sorted(glob.glob("F2/*.tif"))

    A.spot_finder(phase, parB, parA, cell_lines)
