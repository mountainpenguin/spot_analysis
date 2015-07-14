#!/usr/bin/env python

import sys
sys.path.append("/home/miles/Data/Work/MRes/project/lineage-app/lineage/lineage")
from lineage_lib import track
import glob
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import scipy.signal
import scipy.stats
import numpy as np
import json
import datetime


class Analysis(track.Lineage):
    MEAN = 1
    SUM = 2
    MAX = 3
    def __init__(self, debug, method=MEAN):
        self.debug = debug
        self.method = method
        track.Lineage.__init__(self)

    def apply_alignments(self):
        f = 0
        for align in self.alignment:
            cells = self.frames[f].cells
            for c in cells:
                c.mesh = c.mesh - [
                    align[1], align[0], align[1], align[0]
                ]
            f += 1

    def _gettimestamp(self, day, time, *args):
        return datetime.datetime.strptime(
            "{0} {1}".format(day, time),
            "%d.%m.%y %H:%M"
        )

    def _timediff(self, day, time, t0):
        t1 = self._gettimestamp(day, time)
        td = t1 - t0
        s = td.days * 24 * 60 * 60
        s += td.seconds
        m = s // 60
        return m

    def get_timings(self, t0=False):
        timing_data = json.loads(open("timings.json").read())
        timings = timing_data["timings"]
        self.T = []
        if "start" in timing_data:
            t0 = self._gettimestamp(*timing_data["start"])
        else:
            t0 = self._gettimestamp(*timings[0])
        for d1, t1, frames in timings:
            sm = self._timediff(d1, t1, t0)
            for _ in range(frames):
                self.T.append(sm)
                sm += timing_data["pass_delay"]

    def process_fluor(self, img):
        # Gaussian smoothing to denoise
        img = self.denoise(img)
        return img

    def denoise(self, img):
        out = scipy.ndimage.gaussian_filter(img, 2)
        return out

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

        f2 = scipy.misc.imread(self.fluor2[frame_idx])
        f2_denoise = self.process_fluor(f2)

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

        # M = np.column_stack((M_x, M_y))
        i = np.array(range(len(F)))
        bg = f.mean()
        bg2 = f2.mean()

        # get peaks
        m = (F - bg).mean()
        # s = np.std(F)

#        # wavelet peak-finding method
#        peaks = scipy.signal.find_peaks_cwt(
#            F - bg, np.arange(1, 10)
#        )
#        vals = [
#            (i[_p], (F - bg)[_p]) for _p in peaks if (F - bg)[_p] > m
#        ]

        # argrelextrema peak-finding method
        peaks = scipy.signal.argrelextrema(F - bg, np.greater_equal)[0]
        vals = [
            (i[_z], (F - bg)[_z]) for _z in peaks if (F - bg)[_z] > m
        ]

        # check distances between peaks
        if len(vals) > 1:
            modded = [(_i[0] // 10, (_i[0], _i[1])) for _i in vals]
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
                    # average
                    index = int(np.mean([_val[0] for _val in group]))
                    out.append((index, group[0][1]))
                else:
                    out.append(group[0])
            vals = sorted(out, key=lambda x: x[0])

        if self.debug:
            plt.figure()
            ax = plt.subplot(231)
            plt.imshow(p, cmap=plt.cm.gray)
            plt.plot(M_x, M_y, "r-")
            _ = 0
            while _ < xs_l.shape[0]:
                plt.plot(
                    (xs_l[_], xs_r[_]),
                    (ys_l[_], ys_r[_]),
                    "y-"
                )
                _ += 1

            for p in vals:
                x, y = M_x[p[0]], M_y[p[0]]
                plt.plot(
                    x, y, "r*"
                )

            plt.subplot(232, sharex=ax, sharey=ax)
            plt.imshow(f, cmap=plt.cm.hot)
            plt.plot(xs_l, ys_l, "k-")
            plt.plot(xs_r, ys_r, "k-")

            for p in vals:
                x, y = M_x[p[0]], M_y[p[0]]
                plt.plot(
                    x, y, "k*"
                )

            plt.subplot(233, sharex=ax, sharey=ax)
            plt.imshow(f2, cmap=plt.cm.hot)
            plt.plot(xs_l, ys_l, "k-")
            plt.plot(xs_r, ys_r, "k-")

            plt.subplot(234)
            plt.plot(i, F - bg, "k-")
            plt.plot(i, F_unsmoothed - bg, "k-", alpha=0.4)
            plt.plot([i[0], i[-1]], [m, m])
            # plt.plot([i[0], i[-1]], [m + s, m + s], "y-")

            for p in vals:
                plt.plot(
                    p[0], p[1], "r."
                )

            plt.subplot(235)
            plt.plot(i, F2 - bg2, "k-")
            plt.plot(i, F2_unsmoothed - bg2, "k-", alpha=0.4)

            plt.show()

        vals = [
            ((__[0] / i[-1]) * cell.length[0][0], __[1], cell.length[0][0]) for __ in vals
        ]
        return vals

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

    def spot_finder(self, phase, fluor, fluor2, cell_lines):
        self.phase = phase
        self.fluor = fluor
        self.fluor2 = fluor2
        self.get_timings()
        for cell_line in cell_lines:
            L = []
            S = []
            prior = None
            for cell in cell_line:
                L.append(cell.length[0][0])
                if prior:
                    orientation = self.get_orientation(cell, prior)
                else:
                    orientation = True
                spots = self.get_spots(cell, orientation)
                S.append(spots)
                if orientation:
                    prior = cell.mesh[0, 0:2], cell.mesh[-1, 0:2]
                else:
                    prior = cell.mesh[-1, 0:2], cell.mesh[0, 0:2]

            plt.figure()
            plt.subplot(131)
            start = cell_line[0].frame - 1
            end = cell_line[-1].frame - 1
            t = np.array(self.T[start:end + 1])

            # plot cell lengths with cell-centre at 0
            L = np.array(L)

            plt.plot(t, L / 2, "k-")
            plt.plot(t, (L / 2) * -1, "k-")

            # track spot1 and spot2
            # spot 1 is first spot and spot 2 is last spot
#            spot1 = [
#                x[0] - (x[2] / 2) for x in [
#                    y for y in S
#                ]
#            ]
#            print(spot1)
            spot1 = []
            spot2 = []
            i = 0
            for tp in S:
                if len(tp) > 1:
                    spot1.append((t[i], tp[0][0] - (tp[0][2] / 2)))
                    spot2.append((t[i], tp[-1][0] - (tp[-1][2] / 2)))
                else:
                    spot1.append((t[i], np.NaN))
                    spot2.append((t[i], np.NaN))
                i += 1
            spot1 = np.array(spot1)
            spot2 = np.array(spot2)

            plt.plot(spot1[:, 0], spot1[:, 1], "r-")
            plt.plot(spot2[:, 0], spot2[:, 1], "b-")

            i = 0
            for s in S:
                for p in s:
                    midcell = p[2] / 2
                    spot = p[0] - midcell
                    # plot spots as distance from mid-cell
                    plt.plot(
                        t[i],
                        spot,
                        "r*"
                    )
                i += 1
            plt.ylabel("Distance from mid-cell (px)")
            plt.xlabel("Time (min)")

            # dependent analysis
            # 1. subtract the mean from all values
#            y1 = (-L / 2) - ((-L / 2).mean())
#            y2 = spot1[:, 1] - (spot1[:, 1].mean())
            t_analysis1 = t.copy()
            t_analysis2 = t.copy()
            pole1_norm = -(L / 2)
            pole1_norm = pole1_norm - pole1_norm.mean()
            pole2_norm = (L / 2) - ((L / 2).mean())
            spot1_norm = spot1[:, 1] - np.nanmean(spot1[:, 1])
            spot2_norm = spot2[:, 1] - np.nanmean(spot2[:, 1])

            i = len(spot1_norm) - 1
            while i > -1:
                if np.isnan(spot1_norm[i]):
                    t_analysis1 = np.delete(t_analysis1, i)
                    pole1_norm = np.delete(pole1_norm, i)
                    spot1_norm = np.delete(spot1_norm, i)
                if np.isnan(spot2_norm[i]):
                    t_analysis2 = np.delete(t_analysis2, i)
                    pole2_norm = np.delete(pole2_norm, i)
                    spot2_norm = np.delete(spot2_norm, i)
                i -= 1

            plt.subplot(132)
            plt.plot(t_analysis1, pole1_norm, "r--")
            plt.plot(t_analysis1, spot1_norm, "r-")
            plt.plot(t_analysis2, pole2_norm, "b--")
            plt.plot(t_analysis2, spot2_norm, "b-")
            plt.title("Normalised")
            plt.xlabel("Time (min)")

            # 2. get difference
            d1 = spot1_norm - pole1_norm
            d2 = spot2_norm - pole2_norm
            ax = plt.subplot(133)
            plt.plot(t_analysis1, d1, "r-")
            plt.plot(t_analysis2, d2, "b-")
            plt.title("Difference")
            plt.xlabel("Time (min)")

            # 3. calculate regression of d (y = mx + c)
#            r_m, r_c, r_r, r_p, r_s = scipy.stats.linregress([t_analysis, d])
#            print("y = {0:.2f}x + {1:.2f}".format(r_m, r_c))
#            print("r = {0:.2f}, p = {1:.5f}, stderr = {2:.5f}".format(
#                r_r, r_p, r_s
#            ))
#            regress_x = np.linspace(t_analysis[0], t_analysis[-1])
#            regress_y = r_m * regress_x + r_c
#            plt.plot(regress_x, regress_y, "k--")
#            plt.text(0.05, 0.97, r"$y = {0:.4f}x + {1:.4f}$; $r = {2:.4f}$; $p = {3:.6f}$".format(
#                r_m, r_c, r_r, r_p
#            ), transform=ax.transAxes)
#            # 4. H0: m = 0; H1: m != 0

            plt.show()


if __name__ == "__main__":
    if "-d" in sys.argv:
        debug = True
    else:
        debug = False
    A = Analysis(debug=debug)
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
