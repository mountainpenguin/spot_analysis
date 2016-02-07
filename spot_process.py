#!/usr/bin/env python

import glob
import re
import numpy as np
import datetime
import os
import tkinter.filedialog
import sys
import json

import matplotlib.pyplot as plt
for x in plt.rcParams:
    if x.startswith("keymap"):
        plt.rcParams[x] = ""

import matplotlib.gridspec
import matplotlib.patches
import matplotlib.transforms

from lineage_lib import track
from analysis_lib import shared
import spot_plot
import spot_spread


class InteractivePlot(object):
    MODE_ADD = [
        ("ESC", "Discard changes", "escape"),
        ("Enter", "Accept changes", "enter"),
        ("Click", "Add new spot", ""),
    ]
    MODE_REMOVE = [
        ("ESC", "Discard changes", "escape"),
        ("Enter", "Accept changes", "enter"),
        ("Click", "Remove spot", ""),
    ]
    MODE_REPLACE = [
        ("ESC", "Discard changes", "escape"),
        ("Enter", "Accept changes", "enter"),
        ("Click", "Replace spot", ""),
    ]

    def __init__(self, cell_line, lineage_num, noplot=False):
        self.GEN_PLOTS = not noplot

        self.par_path_ind = None
        self.img_plot_targeting_line = None
        self.trace_plot_targeting_lines = []
        self.par_spots = []
        self.par_img_spots = []

        self.T = cell_line[0].T

        self.grid_spec = matplotlib.gridspec.GridSpec(2, 2)
        self.cell_line = cell_line
        self.lineage_num = int(lineage_num)
        self.current_cell = cell_line[0]
        self.current_cell_idx = 0
        self.fig = plt.gcf()
        self.fig.patch.set_alpha(0)
        self.fig.clear()

        self.img_plot = self.fig.add_subplot(self.grid_spec[0, 0])
        self.trace_plot = self.fig.add_subplot(self.grid_spec[1, 0])
        self.status_plot = self.fig.add_subplot(self.grid_spec[0, 1])
        self.par_plot = self.fig.add_subplot(self.grid_spec[1, 1])

        self._decorate()

        self.fig.canvas.mpl_connect("motion_notify_event", self.motion_notify_event)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)
        self.fig.canvas.mpl_connect("pick_event", self.pick_event)
        self.fig.canvas.mpl_connect("button_release_event", self.button_release_event)

        self.draw_cell()
        self.update_par_time_indicator()
        self.MODE = "default"
        self.mode_default()
        plt.show()

    def _deaxis(self, ax=None):
        if not ax:
            ax = plt.gca()

        ax.axis("off")

    def _despine(self, ax=None):
        if not ax:
            ax = plt.gca()

        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

    def _decorate(self):
        self._deaxis(self.img_plot)
        self._despine(self.trace_plot)
        self._deaxis(self.status_plot)
        self._despine(self.par_plot)

    def update_status(self):
        self.status_plot.clear()
        self._deaxis(self.status_plot)
        if self.MODE == "default":
            self.bindings = self.MODE_DEFAULT

        elif self.MODE == "add":
            self.bindings = self.MODE_ADD

        elif self.MODE == "remove":
            self.bindings = self.MODE_REMOVE

        elif self.MODE == "replace":
            self.bindings = self.MODE_REPLACE

        col1 = [
            "Lineage:",
            "Frame:",
            "",
            "Keys:",
        ]
        col2 = [
            str(self.lineage_num),
            str(self.current_cell.frame),
            "",
            "",
        ]
        for k, v, _ in self.bindings:
            col1.append(k)
            col2.append(v)

        txt1 = "\n".join(col1)
        font1 = matplotlib.font_manager.FontProperties()
        font1.set_size("medium")
        font1.set_weight("bold")
        font1.set_family("sans-serif")
        txt2 = "\n".join(col2)
        font2 = font1.copy()
        font2.set_weight("normal")

        txt_params = {
            "transform": self.status_plot.transAxes,
            "horizontalalignment": "left",
            "verticalalignment": "top",
        }

        self.status_plot.text(
            0, 1, txt1,
            fontproperties=font1,
            **txt_params
        )
        self.status_plot.text(
            0.3, 1, txt2,
            fontproperties=font2,
            **txt_params
        )

    def mode_default(self):
        self.MODE = "default"
        # provide instructions on status_plot
        self.fig.suptitle("Default Mode")
        self.update_status()
        self.draw_par()
        self.determine_par_path()
        self.draw_par_path()
        self.redraw()

    def mode_add(self):
        self.MODE = "add"
        # modify suptitle and status
        self.fig.suptitle("Add Mode")
        self.update_status()

        # fade existing Par spots
        for x in self.par_spots:
            x.set_color("b")
            x.set_alpha(0.8)

        # store current Par spots as backup
        self.Par_backup = list(self.get_par())
        self.redraw()

    def mode_remove(self):
        self.MODE = "remove"
        # modify suptitle and status
        self.fig.suptitle("Remove Mode")
        self.update_status()

        # re-colour existing Par spots
        for x in self.par_spots:
            x.set_color("b")

        # backup current Par spots
        self.Par_backup = list(self.get_par())
        self.redraw()

    def mode_replace(self):
        self.MODE = "replace"
        self.fig.suptitle("Replace Mode")
        self.update_status()

        for x in self.par_spots:
            x.set_color("b")

        self.Par_backup = list(self.get_par())
        self.redraw()

    def draw_cell(self):
        img = self.get_img()
        img[np.isnan(img)] = 0
        self.img_plot.imshow(img, cmap=plt.cm.viridis)
        self.img_plot.plot(self.current_cell.mesh[:, 0], self.current_cell.mesh[:, 1], "k-")
        self.img_plot.plot(self.current_cell.mesh[:, 2], self.current_cell.mesh[:, 3], "k-")

        # centre = self.current_cell.centre
        centre = track.Lineage.get_mesh_centre(None, self.current_cell)
        rang = 40
        xmin = centre[0] - rang
        xmax = centre[0] + rang
        ymin = centre[1] - rang
        ymax = centre[1] + rang
        self.img_plot.set_xlim([xmin, xmax])
        self.img_plot.set_ylim([ymin, ymax])

        smoothed = self.get_fluor_smoothed()
        unsmoothed = self.get_fluor_unsmoothed()
        self.trace_plot.plot(
            range(len(unsmoothed)),
            unsmoothed,
            "k-", alpha=0.4, lw=2
        )
        self.trace_plot.plot(
            range(len(smoothed)),
            smoothed,
            "k-", lw=2
        )
        self.redraw()

    def next(self):
        self.current_cell_idx += 1
        try:
            self.current_cell = self.cell_line[self.current_cell_idx]
        except IndexError:
            self.end()

        self.img_plot.clear()
        self.trace_plot.clear()
        self.update_status()
        self.update_par_time_indicator()
        self._decorate()
        self.draw_cell()
        self.mode_default()

    def pick_event(self, event):
        if self.MODE == "remove":
            event.artist.set_color("r")
            # remove spot from self.current_cell.ParB
            idx = 0
            for x in self.current_cell.ParB:
                spot_id = x[0] * x[1]
                if spot_id == event.artist.spot_id:
                    self.current_cell.ParB.pop(idx)
                    break
                idx += 1

            # add red spot on img_plot
            img_x = event.artist.img_x
            img_y = event.artist.img_y
            pb, = self.img_plot.plot(
                img_x, img_y,
                marker="o", ms=10, mec="r", mew=2, mfc="none"
            )
            self.par_img_spots.append(pb)

            # recalculate par lines
            self.determine_par_path()
            self.draw_par_path()

            self.redraw()

    def button_release_event(self, event):
        if self.MODE == "add" and event.button == 1 and event.inaxes == self.trace_plot:
            # add ParB spot
            # update parB paths
            xpos = event.xdata

            x1pos, y1pos = self.current_cell.M_x[xpos], self.current_cell.M_y[xpos]
            pb, = self.img_plot.plot(
                x1pos, y1pos, marker=".", ms=10, mec="r", mew=2, mfc="none"
            )
            self.par_img_spots.append(pb)

            intensity = self.current_cell.parB_fluorescence_smoothed[xpos]
            self.current_cell.ParB.append(
                (xpos, intensity, self.current_cell.length[0][0])
            )
            self.determine_par_path()
            self.draw_par_path()
            pb, = self.trace_plot.plot(xpos, intensity, "r.", ms=10)
            pb.set_picker(5)
            pb.spot_id = xpos * intensity
            pb.img_x = x1pos
            pb.img_y = y1pos
            self.par_spots.append(pb)

            self.redraw()
        if self.MODE == "replace" and event.button == 1 and event.inaxes == self.trace_plot:
            # remove existing ParA spot
            # add new spot
            # update par paths
            xpos = event.xdata
            x1pos, y1pos = self.current_cell.M_x[xpos], self.current_cell.M_y[xpos]
            pb, = self.img_plot.plot(
                x1pos, y1pos, marker=".", ms=10, mec="r", mew=2, mfc="none"
            )
            self.par_img_spots.append(pb)

            intensity = self.current_cell.parA_fluorescence_smoothed[xpos]
            # hack for incorrect parA stuff
            self.current_cell.ParA = (
                xpos,
                intensity + self.current_cell.parA_img.mean(),
                self.current_cell.length[0][0]
            )
            self.determine_par_path()
            self.draw_par_path()
            pb, = self.trace_plot.plot(xpos, intensity, "r.", ms=10)
            self.par_spots.append(pb)
            self.redraw()

    def key_press_event(self, event):
        if self.MODE == "default":
            if event.key not in [x[2] for x in self.bindings]:
                return
            if event.key == "enter":
                self.next()
            elif event.key == "escape":
                self.end(discard=True)
            elif event.key == "a":
                self.mode_add()
            elif event.key == "r":
                self.mode_remove()
            elif event.key == "t":
                self.mode_replace()

        elif self.MODE == "add":
            if event.key == "escape":
                # discard changes
                self.discard()
                self.clear_targeting_lines()
                self.mode_default()
            elif event.key == "enter":
                # accept changes
                self.clear_targeting_lines()
                self.mode_default()
            else:
                print(event.key)
        elif self.MODE == "remove":
            if event.key == "escape":
                # discard changes
                self.discard()
                self.mode_default()
            elif event.key == "enter":
                self.mode_default()
        elif self.MODE == "replace":
            if event.key == "escape":
                self.discard()
                self.clear_targeting_lines()
                self.mode_default()
            elif event.key == "enter":
                self.clear_targeting_lines()
                self.mode_default()

        else:
            if event.key == "escape":
                self.clear_targeting_lines()
                self.mode_default()

    def motion_notify_event(self, event):
        if (self.MODE == "add" or self.MODE == "replace") and event.inaxes == self.trace_plot:
            xpos = event.xdata
            try:
                fluor = self.get_fluor_smoothed()
                ytarget = fluor[xpos]
            except:
                return

            self.clear_targeting_lines()
            _, = self.trace_plot.plot(
                [0, xpos], [ytarget, ytarget], "r-", lw=2
            )
            self.trace_plot_targeting_lines.append(_)
            _, = self.trace_plot.plot(
                [xpos, xpos], [self.trace_plot.get_ylim()[0], ytarget], "r-", lw=2
            )
            self.trace_plot_targeting_lines.append(_)
            _, = self.trace_plot.plot(
                xpos, ytarget, marker=".", mew=2, mec="r", mfc="r"
            )
            self.trace_plot_targeting_lines.append(_)

            x1pos, y1pos = self.current_cell.M_x[xpos], self.current_cell.M_y[xpos]
            self.img_plot_targeting_line, = self.img_plot.plot(
                x1pos, y1pos, marker=".", ms=10, mec="r", mew=2, mfc="none", alpha=0.4
            )
            self.redraw()

    def clear_targeting_lines(self):
        if self.img_plot_targeting_line:
            try:
                self.img_plot.lines.remove(self.img_plot_targeting_line)
            except:
                pass
        self.img_plot_targeting_lines = None

        for l in self.trace_plot_targeting_lines:
            if l:
                try:
                    self.trace_plot.lines.remove(l)
                except:
                    pass
        self.trace_plot_targeting_lines = []

    def draw_par(self):
        for x in self.par_spots:
            try:
                self.trace_plot.lines.remove(x)
            except:
                pass

        for x in self.par_img_spots:
            try:
                self.img_plot.lines.remove(x)
            except:
                pass

        self.par_spots = []
        self.par_img_spots = []

        for par in self.get_par():
            img_x, img_y = self.current_cell.M_x[par[0]], self.current_cell.M_y[par[0]]
            pb, = self.img_plot.plot(img_x, img_y, marker="o", ms=10, mec="b", mew=2, mfc="none")
            self.par_img_spots.append(pb)

            trace_x, trace_y = par[0], par[1]
            # hack for incorrect ParA value
            if self.TARGET == "A":
                trace_y = trace_y - self.get_img().mean()
            pb, = self.trace_plot.plot(trace_x, trace_y, "r.", ms=10)
            pb.set_picker(5)
            pb.spot_id = par[0] * par[1]
            pb.img_x = img_x
            pb.img_y = img_y
            self.par_spots.append(pb)
        self.redraw()

    def draw_par_path(self):
        self.par_plot.clear()
        self._despine(self.par_plot)
        self.update_par_time_indicator()
        spotnum = 1
        for x in self.spots:
            s = x.spots(False)
            self.par_plot.plot(
                s[:, 0], s[:, 1],
                lw=2, marker=".", mec="k",
                ms=10, label="Spot {0}".format(spotnum)
            )
            spotnum += 1

    def update_par_time_indicator(self):
        if self.par_path_ind:
            try:
                self.par_plot.lines.remove(
                    self.par_path_ind
                )
            except:
                pass
        x = self.T[self.current_cell.frame - 1]

        trans = matplotlib.transforms.blended_transform_factory(
            self.par_plot.transData, self.par_plot.transAxes
        )
        self.par_path_ind, = self.par_plot.plot(
            [x, x], [0, 1], "k-",
            alpha=0.4,
            transform=trans
        )

    def redraw(self):
        plt.draw()

    def end(self, discard=False):
        if discard:
            plt.close()
            return

        # backup previous lineage file
        previous_file = "data/cell_lines/lineage{0:02d}.npy".format(self.lineage_num)
        if os.path.exists(previous_file):
            if not os.path.exists("data/cell_lines/backups"):
                os.mkdir("data/cell_lines/backups")
            backup_file = "data/cell_lines/backups/{0}-lineage{1:02d}.npy".format(
                datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M"),
                self.lineage_num
            )
            attempt = 2
            while os.path.exists(backup_file):
                first = backup_file.split(".npy")[0]
                backup_file = "{0}.npy-{1}".format(first, attempt)
                attempt += 1
            print("Backing up previous lineage file to {0}".format(backup_file))
            os.rename(previous_file, backup_file)

        print("Saving new lineage file")
        np.save(
            "data/cell_lines/lineage{0:02d}".format(self.lineage_num),
            self.cell_line
        )

        if self.GEN_PLOTS:
            print("Saving new PDFs")
            spot_plot.plot_images(self.cell_line, int(self.lineage_num))
            spot_plot.plot_graphs(self.cell_line, int(self.lineage_num))

        spot_spread.gen_xl(self.cell_line, int(self.lineage_num))

        plt.close(self.fig)


class InteractivePlotA(InteractivePlot):
    TARGET = "A"
    MODE_DEFAULT = [
        ("ESC", "Discard lineage", "escape"),
        ("Enter", "Next cell", "enter"),
        ("t", "Replace Mode", "t"),
    ]
    def determine_par_path(self):
        # get ParA path
        self.spots = shared.get_parA_path(self.cell_line, self.T)

    def get_img(self):
        return self.current_cell.parA_img

    def get_fluor_smoothed(self):
        return self.current_cell.parA_fluorescence_smoothed

    def get_fluor_unsmoothed(self):
        return self.current_cell.parA_fluorescence_unsmoothed

    def get_par(self):
        return [self.current_cell.ParA]

    def discard(self):
        self.current_cell.ParA = list(self.Par_backup)[0]


class InteractivePlotB(InteractivePlot):
    TARGET = "B"
    MODE_DEFAULT = [
        ("ESC", "Discard lineage", "escape"),
        ("Enter", "Next cell", "enter"),
        ("a", "Add Mode", "a"),
        ("r", "Remove Mode", "r"),
    ]
    def determine_par_path(self):
        self.spots = shared.get_parB_path(self.cell_line, self.T)

    def get_img(self):
        return self.current_cell.parB_img

    def get_fluor_smoothed(self):
        return self.current_cell.parB_fluorescence_smoothed

    def get_fluor_unsmoothed(self):
        return self.current_cell.parB_fluorescence_unsmoothed

    def get_par(self):
        return self.current_cell.ParB

    def discard(self):
        self.current_cell.ParB = list(self.Par_backup)


def process(f, noplot=False, lineage_num=None, target="ParB"):
    cell_line = np.load(f)
    if not lineage_num:
        lineage_num = re.search("lineage(\d+)\.npy", f).group(1)

    if target == "ParB":
        InteractivePlotB(cell_line, lineage_num, noplot=noplot)
    elif target == "ParA":
        InteractivePlotA(cell_line, lineage_num, noplot=noplot)
    else:
        print("Not implemented")


def get_path(wildcard=None):
    root = tkinter.Tk()
    root.withdraw()
    path = tkinter.filedialog.askdirectory()
    return path


def main(noplot=False, wanted=True, target="ParB"):
    if wanted:
        wantedfile = json.loads(open(
            "../../wanted.json"
        ).read())
        base, subdir = os.path.split(os.getcwd())
        base = os.path.basename(base)
        if base in wantedfile and subdir in wantedfile[base]:
            wantedlineages = [int(x) for x in wantedfile[base][subdir]]
        else:
            wantedlineages = None
    else:
        wantedlineages = range(100)

    if wantedlineages is None:
        print("No desired lineages in this directory")
        return

    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    if not targetfiles:
        path = get_path()
        if path and os.path.exists(path):
            os.chdir(path)
            targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
        else:
            print("No files found")
            return
    for targetfile in targetfiles:
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        if lineage_num in wantedlineages:
            process(targetfile, noplot=noplot, lineage_num=lineage_num, target=target)


if __name__ == "__main__":
    if "-w" in sys.argv:
        # ignore wanted file
        wanted = False
    else:
        wanted = True

    if "A" in sys.argv or "a" in sys.argv:
        target = "ParA"
    else:
        target = "ParB"
    main(wanted=wanted, target=target)
