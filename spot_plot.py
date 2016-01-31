#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.gridspec
import glob
import numpy as np
import re
import seaborn as sns
sns.set_style("white")

from analysis_lib import shared


def _plot_mesh(ax, mesh):
    ax.plot(mesh[:, 0], mesh[:, 1], "w-")
    ax.plot(mesh[:, 2], mesh[:, 3], "w-")


def _plot_limits(ax, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _deaxis(ax=None):
    if not ax:
        ax = plt.gca()
    ax.axis("off")


def _despine(ax=None):
    if not ax:
        ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def _fmt_img(ax, mesh, xlim, ylim):
    _plot_mesh(ax, mesh)
    _plot_limits(ax, xlim, ylim)
    _deaxis(ax)


def plot_images(cell_line, lineage_num):
    grid_width = len(cell_line)
    fig = plt.figure(figsize=(grid_width * 5, 20))
    fig.patch.set_alpha(0)
    gs = matplotlib.gridspec.GridSpec(4, grid_width)
    sp_num = 0
    for cell in cell_line:
        # xmin, xmax for each image
        # centre +/- 40
        xmin = cell.centre[0] - 40
        xmax = cell.centre[0] + 40
        ymin = cell.centre[1] - 40
        ymax = cell.centre[1] + 40

        # plot phase in gray with white mesh
        ax = fig.add_subplot(gs[0, sp_num:sp_num + 1])
        ax.imshow(cell.phase_img, cmap=plt.cm.gray)
        _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        # plot ParA in RGB red with white mesh
        ax = fig.add_subplot(gs[1, sp_num:sp_num + 1])
        parA_img = np.dstack((
            cell.parA_img / np.nanmax(cell.parA_img),
            np.zeros(cell.parA_img.shape),
            np.zeros(cell.parA_img.shape)
        ))
        ax.imshow(parA_img)
        # plot ParA focus as white spot
        x, y = cell.M_x[cell.ParA[0]], cell.M_y[cell.ParA[0]]
        plt.plot(x, y, "wo", ms=10)
        _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        # plot ParB in RGB green with white mesh
        ax = fig.add_subplot(gs[2, sp_num:sp_num + 1])
        parB_img = np.dstack((
            np.zeros(cell.parB_img.shape),
            cell.parB_img / np.nanmax(cell.parB_img),
            np.zeros(cell.parB_img.shape)
        ))
        ax.imshow(parB_img)
        # plot ParB foci as white spots
        for parB in cell.ParB:
            x, y = cell.M_x[parB[0]], cell.M_y[parB[0]]
            plt.plot(x, y, "wo", ms=10)
        _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        # plot line-scan profile
        ax = fig.add_subplot(gs[3, sp_num:sp_num + 1])
        # X=position, Y=intensity
        # smoothed in black
        # unsmoothed in gray
        position_vals = range(len(cell.parB_fluorescence_smoothed))
        ax.plot(position_vals, cell.parB_fluorescence_smoothed, "k-", lw=5)
        ax.plot(position_vals, cell.parB_fluorescence_unsmoothed, "k-", alpha=0.4, lw=5)
        # plot ParB foci as red spots along smoothed line
        for parB in cell.ParB:
            plt.plot(parB[0], parB[1], "r.", ms=30)

        _despine(ax)
        plt.xlabel("Position")
        plt.ylabel("Intensity")

        sp_num += 1

    # save figure
    plt.tight_layout()
    plt.savefig("data/image-lineage{0:02d}.pdf".format(lineage_num))
    plt.close()


def plot_graphs(cell_line, lineage_num):
    fig = plt.figure(figsize=(20, 10))
    gs = matplotlib.gridspec.GridSpec(2, 3)
    fig.patch.set_alpha(0)

    spots_ParA = []
    L = []
    T = shared.get_timings()
    for cell in cell_line:
        spots_ParA.append(cell.ParA)
        L.append(cell.length[0][0])
    L = np.array(L)

    start = cell_line[0].frame - 1
    end = cell_line[-1].frame - 1
    t = np.array(T[start:end + 1])

    ax = fig.add_subplot(gs[0, 0])
    _despine(ax)
    ax.set_title("ParA")
    ax.plot(t, L / 2, "k-", lw=2)
    ax.plot(t, -(L / 2), "k-", lw=2)

    parAs = {}
    i = 0
    for s in spots_ParA:
        midcell = s[2] / 2
        spot = s[0] - midcell
        parAs[t[i]] = spot
        i += 1
    parAs = np.array(sorted(parAs.items()))
    ax.plot(
        parAs[:, 0], parAs[:, 1],
        lw=2, marker=".", mec="k", ms=10
    )
    ax.set_xlabel(r"Time (min)")
    ax.set_ylabel(r"Distance from mid-cell (px)")

    ax_parA = fig.add_subplot(gs[1, 0])
    _despine(ax_parA)
    ax_parA.set_title("ParA maximum")
    ax_parA.plot(
        parAs[:, 0], (L / 2) - parAs[:, 1],
        marker=".", lw=2, mec="k", ms=10,
        label="Distance from top pole"
    )
    ax_parA.plot(
        parAs[:, 0], (L / 2) + parAs[:, 1],
        marker=".", lw=2, mec="k", ms=10,
        label="Distance from bottom pole"
    )
    ax_parA.set_xlabel(r"Time (min)")
    ax_parA.set_ylabel(r"Distance (px)")

    ax = fig.add_subplot(gs[0, 1])
    _despine(ax)
    ax.set_title("ParB")
    ax.plot(t, L / 2, "k-", lw=2, label="Cell poles")
    ax.plot(t, -(L / 2), "k-", lw=2)
    spots_ParB = shared.get_parB_path(cell_line, T)
    spotnum = 1
    for x in spots_ParB:
        s = x.spots(False)
        ax.plot(
            s[:, 0], s[:, 1],
            lw=2, marker=".", markeredgecolor="k", ms=10,
            label="Spot {0}".format(spotnum)
        )

        dparA = []
        for spot in s:
            parApos = parAs[np.where(parAs[:, 0] == spot[0])][0, 1]
            dpA = spot[1] - parApos
            dparA.append(dpA)

        x.parA_d = dparA
        x.parA_dmean = np.mean(dparA)
        x.parA_dsem = np.std(dparA) / np.sqrt(len(dparA))
        x.spotnum = spotnum
        x.intensity_mean = s[:, 2].mean()

        spotnum += 1
    ax.set_ylabel(r"Distance from mid-cell (px)")
    ax.set_xlabel(r"Time (min)")

    ax.legend(bbox_to_anchor=(1.35, 1))

    filtered = [x for x in spots_ParB if len(x) > 4]
    if len(filtered) > 0:
        ax_parB_closest = fig.add_subplot(gs[1, 1], sharex=ax_parA, sharey=ax_parA)
        dmin = min(
            filtered,
            key=lambda x: np.abs(x.parA_dmean)
        )
        ax_parB_closest.set_title("ParB spot {0} (dmin)".format(dmin.spotnum))
        _despine(ax_parB_closest)
        s = dmin.spots(False)
        dpol_t = (dmin.len() / 2) - s[:, 1]
        dpol_b = (dmin.len() / 2) + s[:, 1]

        ax_parB_closest.plot(
            s[:, 0], dpol_t,
            marker=".", lw=2, mec="k", ms=10,
            label="Distance from top pole"
        )
        ax_parB_closest.plot(
            s[:, 0], dpol_b,
            marker=".", lw=2, mec="k", ms=10,
            label="Distance from bottom pole"
        )
        ax_parB_closest.plot(
            s[:, 0], dmin.parA_d,
            marker=".", lw=2, ms=10,
            label="Distance from ParA focus"
        )
        ax_parB_closest.plot(
            ax_parB_closest.get_xlim(),
            [0, 0],
            "k--"
        )

        imax = max(filtered, key=lambda x: x.intensity_mean)
        if imax.id != dmin.id:
            ax = fig.add_subplot(gs[1, 2], sharex=ax_parB_closest, sharey=ax_parA)
            s = imax.spots(False)
            dpol_t = (imax.len() / 2) - s[:, 1]
            dpol_b = (imax.len() / 2) + s[:, 1]
            ax.plot(s[:, 0], dpol_t, marker=".", lw=2, mec="k", ms=10, label="Distance from top pole")
            ax.plot(s[:, 0], dpol_b, marker=".", lw=2, mec="k", ms=10, label="Distance from bottom pole")
            ax.plot(s[:, 0], imax.parA_d, marker=".", lw=2, mec="k", ms=10, label="Distance from ParA focus")

            ax.plot(ax.get_xlim(), [0, 0], "k--")

            _despine(ax)
            ax.set_title("ParB Spot {0} (imax)".format(imax.spotnum))
            ax.set_ylabel(r"Distance (px)")
            ax.set_xlabel("Time (min)")
            ax.legend(bbox_to_anchor=(0.8, 1.35))
        else:
            ax_parB_closest.legend(bbox_to_anchor=(1.65, 1))
    else:
        ax_parA.legend(bbox_to_anchor=(1.65, 1))

    plt.tight_layout()
    plt.savefig("data/data-lineage{0:02d}.pdf".format(lineage_num))
    plt.close()


if __name__ == "__main__":
    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    for targetfile in targetfiles:
        cell_line = np.load(targetfile)
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        plot_images(cell_line, lineage_num)
        plot_graphs(cell_line, lineage_num)
        print("Generated plots for lineage {0}".format(lineage_num))
