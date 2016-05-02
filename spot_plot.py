#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.transforms
import glob
import numpy as np
import re
import seaborn as sns
sns.set_style("white")
import sys
import json
import os

from analysis_lib import shared
from lineage_lib import track
from lineage_lib import poles

PX = 0.12254


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
    ax.xaxis.set_label_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")


def _fmt_img(ax, mesh, xlim, ylim):
    _plot_mesh(ax, mesh)
    _plot_limits(ax, xlim, ylim)
    _deaxis(ax)


def plot_images(cell_line, lineage_num, plot_parA=True):
    grid_width = len(cell_line)
    fig = plt.figure(figsize=(grid_width * 5, 20))
    fig.patch.set_alpha(0)
    gs = matplotlib.gridspec.GridSpec(4, grid_width)
    sp_num = 0

    parA_row = 1
    parB_row = 2
    linescan_row = 3

    if not plot_parA:
        parB_row = 1
        linescan_row = 2

    for cell in cell_line:
        # xmin, xmax for each image
        # centre +/- 40
        cell.centre = track.Lineage.get_mesh_centre(None, cell)
        xmin = cell.centre[0] - 40
        xmax = cell.centre[0] + 40
        ymin = cell.centre[1] - 40
        ymax = cell.centre[1] + 40

        if xmin < 0:
            xmin = 0
            xmax = 80
        elif xmax >= cell.phase_img.shape[0]:
            xmin = cell.phase_img.shape[0] - 81
            xmax = cell.phase_img.shape[0] - 1

        if ymin < 0:
            ymin = 0
            ymax = 80
        elif ymax >= cell.phase_img.shape[0]:
            ymin = cell.phase_img.shape[1] - 81
            ymax = cell.phase_img.shape[1] - 1

        xs = np.concatenate([
            cell.mesh[:, 0],
            cell.mesh[:, 2]
        ])
        ys = np.concatenate([
            cell.mesh[:, 1],
            cell.mesh[:, 3]
        ])
        xdis = xs - xmin
        if np.any(xdis < 0):
            xmin += xdis.min() - 5
        xdas = xmax - xs
        if np.any(xdas < 0):
            xmax += -xdas.min() + 5

        ydis = ys - ymin
        if np.any(ydis < 0):
            ymin += ydis.min() - 5
        ydas = ymax - ys
        if np.any(ydas < 0):
            ymax += -ydas.min() + 5

        # plot phase in gray with white mesh
        ax = fig.add_subplot(gs[0, sp_num:sp_num + 1])
        ax.imshow(cell.phase_img, cmap=plt.cm.gray)
        _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        if plot_parA:
            # plot ParA in RGB red with white mesh
            ax = fig.add_subplot(gs[parA_row, sp_num:sp_num + 1])
            red_chan = cell.parA_img_bg / np.nanmax(cell.parA_img_bg[ymin:ymax, xmin:xmax])
            red_chan[red_chan > 1] = 1
            parA_img = np.dstack((
                red_chan,
                np.zeros(cell.parA_img.shape),
                np.zeros(cell.parA_img.shape)
            ))
            ax.imshow(parA_img)

            # plot ParA focus as white spot
            x, y = cell.M_x[cell.ParA[0]], cell.M_y[cell.ParA[0]]
            plt.plot(x, y, "wo", ms=10)
            _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        # plot ParB in RGB green with white mesh
        ax = fig.add_subplot(gs[parB_row, sp_num:sp_num + 1])
        green_chan = cell.parB_img_bg / np.nanmax(cell.parB_img_bg[ymin:ymax, xmin:xmax])
        green_chan[green_chan > 1] = 1
        parB_img = np.dstack((
            np.zeros(cell.parB_img.shape),
            green_chan,
            np.zeros(cell.parB_img.shape)
        ))
        ax.imshow(parB_img)
        # plot ParB foci as white spots
        for parB in cell.ParB:
            x, y = cell.M_x[parB[0]], cell.M_y[parB[0]]
            plt.plot(x, y, "wo", ms=10)
        _fmt_img(ax, cell.mesh, (xmin, xmax), (ymax, ymin))

        # plot line-scan profile
        ax = fig.add_subplot(gs[linescan_row, sp_num:sp_num + 1])
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
        ax.set_xlabel("Position")
        ax.set_ylabel("Intensity")
        ax.patch.set_alpha(0)

        sp_num += 1

    # save figure
    plt.tight_layout()
    fn = "data/image-lineage{0:02d}.pdf".format(lineage_num)
    plt.savefig(fn)
    print("Saved image file to {0}".format(fn))
    plt.close()


def decorate_daughters(cell_line, lineage, ax, pad=10, labels=None):
    parent_cell = cell_line[-1]

    # plot approximate division site
    children_ids = parent_cell.children

    if children_ids:
        # get info for id
        child1 = lineage.frames.cell(children_ids[1])
        last_child1 = lineage.frames.cell(children_ids[1])
        while type(last_child1.children) is str:
            # iterate to last child id?
            last_child1 = lineage.frames.cell(last_child1.children)

        if labels:
            child1_num = labels[last_child1.id]
        else:
            child1_num = "?"
        child2 = lineage.frames.cell(children_ids[0])
        last_child2 = lineage.frames.cell(children_ids[0])
        while type(last_child2.children) is str:
            last_child2 = lineage.frames.cell(last_child2.children)

        if labels:
            child2_num = labels[last_child2.id]
        else:
            child2_num = "?"

        ldiff = (child1.length[0][0] * PX + child2.length[0][0] * PX -
                 parent_cell.length[0][0] * PX)

        # determine which pole each child corresponds to
        # for cell1/cell2 assignment
        parent_pupper = parent_cell.mesh[0]

        child1_pupper = child1.mesh[0]
        child1_plower = child1.mesh[-1]

        child2_pupper = child2.mesh[0]
        child2_plower = child2.mesh[-1]

        child1_dist = np.min([
            np.abs(np.sum(parent_pupper - child1_pupper)),
            np.abs(np.sum(parent_pupper - child1_plower)),
        ])
        child2_dist = np.min([
            np.abs(np.sum(parent_pupper - child2_pupper)),
            np.abs(np.sum(parent_pupper - child2_plower)),
        ])
        if child1_dist > child2_dist:
            child1 = lineage.frames.cell(children_ids[0])
            child2 = lineage.frames.cell(children_ids[1])
            _ = str(child1_num)
            child1_num = str(child2_num)
            child2_num = str(_)

        trans = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        width = 0.05
        shared_params = {
            "width": width,
            "boxstyle": matplotlib.patches.BoxStyle.Round4(
                pad=0, rounding_size=0.02,
            ),
            "fill": False,
            "edgecolor": "k",
            "linewidth": 2,
            "transform": trans,
        }

        lowerleft_x = 0.93
        # lowest point of last cell - ldiff/2
        # lowest point = -(L[-1] / 2 )
        lowerleft_y1 = -(parent_cell.length[0][0] * PX / 2) - (ldiff / 2)
        cell1 = matplotlib.patches.FancyBboxPatch(
            (lowerleft_x, lowerleft_y1),
            height=child1.length[0][0] * PX,
            **shared_params
        )
        # highest point of child1
        # i.e. lowerleft_y1 + height
        lowerleft_y2 = lowerleft_y1 + child1.length[0][0] * PX
        cell2 = matplotlib.patches.FancyBboxPatch(
            (lowerleft_x, lowerleft_y2),
            height=child2.length[0][0] * PX,
            **shared_params
        )
        ax.add_patch(cell1)
        ax.add_patch(cell2)

        # add text
        text_params = {
            "transform": trans,
            "ha": "center",
            "va": "center",
            "fontsize": 10,
        }
        ax.text(
            lowerleft_x + width / 2,
            lowerleft_y1 + child1.length[0][0] * PX / 2,
            child1_num,
            **text_params
        )

        ax.text(
            lowerleft_x + width / 2,
            lowerleft_y2 + child2.length[0][0] * PX / 2,
            child2_num,
            **text_params
        )

        ylim = np.max([
            child1.length[0][0] * PX + child2.length[0][0] * PX,
            parent_cell.length[0][0] * PX
        ]) + 2

        # add 10% to xlim
        xlimadd = (cell_line[0].t[-1] - cell_line[0].t[0]) / 10
        _plot_limits(
            ax,
            (
                cell_line[0].t[0] - 8,
                cell_line[0].t[-1] + 2 + pad + width + xlimadd
            ),
            (
                -(ylim / 2),
                ylim / 2
            )
        )
    else:
        _plot_limits(
            ax,
            (
                min(cell_line[0].t) - 8,
                max(cell_line[0].t) + 7
            ),
            (
                -(parent_cell.length * PX / 2) - 1,
                parent_cell.length * PX / 2 + 1
            )
        )


def plot_graphs_parB_only(cell_line, lineage_num, ax_parB=None, save=True):
    L = np.array([x.length[0][0] * PX for x in cell_line])
    T = shared.get_timings()
    t = np.array(T[cell_line[0].frame - 1:cell_line[-1].frame])

    lineage = track.Lineage()

    if not ax_parB:
        fig = plt.figure(figsize=(20 / 3, 5))
        ax_parB = fig.add_subplot(111)
        _despine(ax_parB)
        ax_parB.set_title("ParB")
        ax_parB.set_ylabel(r"Distance from mid-cell (px)")
        ax_parB.set_xlabel(r"Time (min)")

    ax_parB.plot(t, L / 2, "k-", lw=2, label="Cell poles")
    ax_parB.plot(t, -(L / 2), "k-", lw=2)

    spots_ParB = shared.get_parB_path(cell_line, T, lineage_num)
    spotnum = 1
    colourwheel = sns.color_palette(n_colors=len(spots_ParB))
    for x in spots_ParB:
        colour = colourwheel[spotnum - 1]
        s = x.spots(False)
        if hasattr(x, "split_parent") and x.split_parent:
            for y in spots_ParB:
                if y.spot_ids[-1] == x.split_parent:
                    textra = y.timing[-1:]
                    timings = np.concatenate([textra, s["timing"]])
                    posextra = y.position[-1:]
                    positions = np.concatenate([posextra, s["position"]])
                    break
        else:
            timings = s["timing"]
            positions = s["position"]

        ax_parB.plot(
            timings, positions,
            lw=2, marker=".",
            mec="k", ms=10,
            label="Spot {0}".format(spotnum),
            color=colour
        )
        spotnum += 1

    lines = ax_parB.lines[::-1]
    for l in lines:
        ax_parB.lines.remove(l)
        ax_parB.add_line(l)

    decorate_daughters(cell_line, lineage, ax_parB, pad=5)

    if save:
        ax_parB.legend(bbox_to_anchor=(1.35, 1))
        ax_parB.patch.set_alpha(0)

        plt.tight_layout()
        fn = "data/data-lineage{0:02d}.pdf".format(lineage_num)
        plt.savefig(fn)
        print("Saved data file to {0}".format(fn))
        plt.close()


def plot_graphs(cell_line, lineage_num, num_plots=5, parA_heatmap=None, save=True, labels=None):
    lineage = track.Lineage()
    if save:
        fig = plt.figure(figsize=(20, 10))
        gs = matplotlib.gridspec.GridSpec(2, 3)
        fig.patch.set_alpha(0)

    spots_ParA = []
    traces_ParA = []
    L = []
    T = shared.get_timings()
    for cell in cell_line:
        spots_ParA.append(cell.ParA)
        traces_ParA.append(cell.parA_fluorescence_smoothed[::-1])
        L.append(cell.length[0][0] * PX)
    L = np.array(L)

    traces_ParA = np.concatenate(traces_ParA)
    max_parA_intensity = traces_ParA.max()

    start = cell_line[0].frame - 1
    end = cell_line[-1].frame - 1
    t = np.array(T[start:end + 1])

    if save:
        parA_heatmap = fig.add_subplot(gs[0, 0])
        _despine(parA_heatmap)
    # use Rectangles to generate a heatmap
    i = 0
    cmapper = plt.cm.get_cmap("afmhot")
    for cell in cell_line:
        trace = cell.parA_fluorescence_smoothed
        l = cell.length[0][0] * PX
        x0 = t[i] - 8
        y0 = -(l / 2)
        increment = l / len(trace)
        colours = cmapper(trace / max_parA_intensity)
        i2 = 0
        for _t in trace:
            if _t < 0:
                _t = 0
            r = matplotlib.patches.Rectangle(
                (x0, y0),
                width=15,
                height=increment,
                facecolor=colours[i2],
                edgecolor="none",
            )
            parA_heatmap.add_patch(r)
            y0 += increment
            i2 += 1
        i += 1

    if save:
        parA_heatmap.set_ylabel(r"Distance from mid-cell (px)")
        parA_heatmap.set_xlabel(r"Time (min)")
        parA_heatmap.set_title("ParA")

    poledict = poles.PoleAssign(lineage.frames).assign_poles()
    if poledict[cell_line[0].id] is None:
        parA_heatmap.plot(t, L / 2, "r-", lw=2)
        parA_heatmap.plot(t, -(L / 2), "r-", lw=2)
    else:
        parA_heatmap.plot(t, L / 2, "k-", lw=2)
        parA_heatmap.plot(t, -(L / 2), "b-", lw=2)

    parAs = {}
    i = 0
    for s in spots_ParA:
        midcell = s[2] / 2
        spot = s[0] - midcell
        parAs[t[i]] = spot
        i += 1
    parAs = np.array(sorted(parAs.items()))

    if num_plots >= 2:
        parA_heatmap.plot(
            parAs[:, 0], parAs[:, 1],
            lw=2, marker=".", mec="k", ms=10
        )

    decorate_daughters(cell_line, lineage, parA_heatmap, labels=labels)
    parA_heatmap.patch.set_alpha(0)

    if num_plots >= 3:
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
        ax_parA.patch.set_alpha(0)

    if num_plots >= 2:
        parB_midcell = fig.add_subplot(gs[0, 1])
        _despine(parB_midcell)
        parB_midcell.set_title("ParB")
        parB_midcell.plot(t, L / 2, "k-", lw=2, label="Cell poles")
        parB_midcell.plot(t, -(L / 2), "k-", lw=2)

    spots_ParB = shared.get_parB_path(cell_line, T, lineage_num)
    spotnum = 1
    if len(spots_ParB) == 1:
        n_colors = 2
    else:
        n_colors = len(spots_ParB)
    colourwheel = sns.color_palette(n_colors=n_colors)
    for x in spots_ParB:
        colour = colourwheel[spotnum - 1]
        s = x.spots(False)
        if hasattr(x, "split_parent") and x.split_parent:
            for y in spots_ParB:
                if y.spot_ids[-1] == x.split_parent:
                    textra = y.timing[-1:]
                    timings = np.concatenate([textra, s["timing"]])
                    posextra = np.array(y.position[-1:])
                    positions = np.concatenate([posextra * PX, s["position"] * PX])
                    break
        else:
            timings = s["timing"]
            positions = s["position"] * PX

        if num_plots >= 2:
            ax_target = parB_midcell
            label = "Spot {0}".format(spotnum)
        else:
            ax_target = parA_heatmap
            colour = colourwheel[1]
            label = "ParB"

        ax_target.plot(
            timings, positions,
            lw=2, marker=".", markeredgecolor="k", ms=10,
            label=label,
            color=colour
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
        x.intensity_mean = s["intensity"].mean()

        spotnum += 1

    if num_plots >= 2:
        lines = parB_midcell.lines[::-1]
        for l in lines:
            parB_midcell.lines.remove(l)
            parB_midcell.add_line(l)

        parB_midcell.set_ylabel(r"Distance from mid-cell (px)")
        parB_midcell.set_xlabel(r"Time (min)")

        decorate_daughters(cell_line, lineage, parB_midcell, pad=5)

        parB_midcell.legend(bbox_to_anchor=(1.35, 1))
        parB_midcell.patch.set_alpha(0)
    else:
        if save:
            parA_heatmap.legend(
                [parA_heatmap.lines[-1]], [parA_heatmap.lines[-1].get_label()],
                bbox_to_anchor=(1.2, 1)
            )

    if num_plots >= 4:
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
            dpol_t = (dmin.len() / 2) - s["position"]
            dpol_b = (dmin.len() / 2) + s["position"]

            ax_parB_closest.plot(
                s["timing"], dpol_t,
                marker=".", lw=2, mec="k", ms=10,
                label="Distance from top pole"
            )
            ax_parB_closest.plot(
                s["timing"], dpol_b,
                marker=".", lw=2, mec="k", ms=10,
                label="Distance from bottom pole"
            )
            ax_parB_closest.plot(
                s["timing"], dmin.parA_d,
                marker=".", lw=2, ms=10,
                label="Distance from ParA focus"
            )
            ax_parB_closest.plot(
                ax_parB_closest.get_xlim(),
                [0, 0],
                "k--"
            )
            ax_parB_closest.patch.set_alpha(0)

            if num_plots >= 5:
                imax = max(filtered, key=lambda x: x.intensity_mean)
                if imax.id != dmin.id:
                    ax_parB_highest = fig.add_subplot(gs[1, 2], sharex=ax_parB_closest, sharey=ax_parA)
                    s = imax.spots(False)
                    dpol_t = (imax.len() / 2) - s["position"]
                    dpol_b = (imax.len() / 2) + s["position"]
                    ax_parB_highest.plot(s["timing"], dpol_t, marker=".", lw=2, mec="k", ms=10, label="Distance from top pole")
                    ax_parB_highest.plot(s["timing"], dpol_b, marker=".", lw=2, mec="k", ms=10, label="Distance from bottom pole")
                    ax_parB_highest.plot(s["timing"], imax.parA_d, marker=".", lw=2, mec="k", ms=10, label="Distance from ParA focus")

                    ax_parB_highest.plot(ax_parB_highest.get_xlim(), [0, 0], "k--")

                    _despine(ax_parB_highest)
                    ax_parB_highest.set_title("ParB Spot {0} (imax)".format(imax.spotnum))
                    ax_parB_highest.set_ylabel(r"Distance (px)")
                    ax_parB_highest.set_xlabel("Time (min)")
                    ax_parB_highest.patch.set_alpha(0)
                    ax_parB_highest.legend(bbox_to_anchor=(0.8, 1.35))
                else:
                    ax_parB_closest.legend(bbox_to_anchor=(1.65, 1))
            else:
                ax_parB_closest.legend(bbox_to_anchor=(1.65, 1))
        else:
            ax_parA.legend(bbox_to_anchor=(1.65, 1))

    if save:
        plt.tight_layout()
        fn = "data/data-lineage{0:02d}.pdf".format(lineage_num)
        plt.savefig(fn)
        print("Saved data file to {0}".format(fn))


def main():
    if "-w" in sys.argv:
        wantedlineages = range(1000)
    else:
        wantedfile = json.loads(open(
            "../../wanted.json"
        ).read())
        base, subdir = os.path.split(os.getcwd())
        base = os.path.basename(base)
        if base in wantedfile and subdir in wantedfile[base]:
            wantedlineages = [int(x) for x in wantedfile[base][subdir]]
        else:
            wantedlineages = None

    num_plots = 5
    for x in sys.argv:
        if "-n" in x and x != "-nograph":
            num_plots = int(x.split("-n")[1])
            break

    if wantedlineages is None:
        print("No desired lineages in this directory")
        return

    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    for targetfile in targetfiles:
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        if lineage_num in wantedlineages:
            cell_line = np.load(targetfile)
            if "-b" in sys.argv:
                if "-noimage" not in sys.argv:
                    plot_images(cell_line, lineage_num, plot_parA=False)
                if "-nograph" not in sys.argv:
                    plot_graphs_parB_only(cell_line, lineage_num)
            else:
                if "-noimage" not in sys.argv:
                    plot_images(cell_line, lineage_num)
                if "-nograph" not in sys.argv:
                    plot_graphs(cell_line, lineage_num, num_plots=num_plots)
            print("Generated plots for lineage {0}".format(lineage_num))

if __name__ == "__main__":
    main()
