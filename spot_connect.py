#!/usr/bin/env python

import sys
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import seaborn as sns
sns.set_style("white")

from analysis_lib import shared


class SpotPicker(object):
    def __init__(self, ID, spot_marker, spot_line, xpos, ypos, colour):
        self.ID = ID
        self.spot_marker = spot_marker
        self.spot_line = spot_line
        self.xpos = xpos
        self.ypos = ypos
        self.colour = colour


class SpotStorage(object):
    def __init__(self):
        self.spots = {}

    def add(self, spot_marker, spot_line, xpos, ypos, colour):
        ID = "".join([random.choice(string.ascii_letters) for x in range(20)])
        self.spots[ID] = SpotPicker(
            ID, spot_marker, spot_line, xpos, ypos, colour
        )
        return ID

    def get(self, ID):
        return self.spots[ID]

    def set(self, ID, spot_info):
        self.spots[ID] = spot_info


class Connector(object):
    def __init__(self, cell_line, lineage_num):
        self.cell_line = cell_line
        self.lineage_num = lineage_num
        self.T = cell_line[0].T
        self.t = cell_line[0].t
        self.spots = shared.get_parB_path(self.cell_line, self.T)
        self.spot_storage = SpotStorage()

        self.plot_setup()

        self.plot_parB()

        plt.show()

    def plot_setup(self):
        self.fig = plt.gcf()
        self.fig.patch.set_alpha(0)
        self.fig.clear()

        self.fig.canvas.mpl_connect("pick_event", self.pick_event)
        self.fig.canvas.mpl_connect("motion_notify_event", self.motion_notify_event)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)

        self.PAR_LINE = None
        self.MODE = 0

        self.par_plot = self.fig.add_subplot(111)
        self.par_plot.spines["right"].set_color("none")
        self.par_plot.spines["top"].set_color("none")
        self.par_plot.xaxis.set_ticks_position("bottom")
        self.par_plot.yaxis.set_ticks_position("left")

    def plot_parB(self):
        colourwheel = sns.color_palette("husl", len(self.spots))
        self.par_plot.clear()
        spotnum = 1
        for x in self.spots:
            colour = colourwheel[spotnum - 1]
            s = x.spots(False)
            index = 0
            for spot in s:
                # plot selectable plot marker
                spot_marker, = self.par_plot.plot(
                    spot[0], spot[1],
                    marker=".", mec="k",
                    ms=10, color=colour
                )
                spot_marker.set_picker(5)
                spot_line = None

                if index < len(s) - 1:
                    # connect spots
                    next_spot = s[index + 1]
                    spot_line, = self.par_plot.plot(
                        [spot[0], next_spot[0]],
                        [spot[1], next_spot[1]],
                        color=colour
                    )

                spot_marker_id = self.spot_storage.add(
                    spot_marker,
                    spot_line,
                    spot[0],
                    spot[1],
                    colour
                )
                spot_marker.marker_id = spot_marker_id

                index += 1
            spotnum += 1

    def pick_event(self, event):
        if not self.MODE:
            event.artist.set_color("r")
            spot_info = self.spot_storage.get(event.artist.marker_id)
            if spot_info.spot_line:
                self.MODE = 1
                self.PAR_SELECTED = event.artist.marker_id
                # remove line
                self.par_plot.lines.remove(spot_info.spot_line)
                # new line connecting to mouse
            plt.draw()
        else:
            # connect to new spot
            # check forward in time!
            print("todo")

    def motion_notify_event(self, event):
        if self.MODE == 1:
            try:
                self.par_plot.lines.remove(self.PAR_LINE)
            except:
                pass

            spot_info = self.spot_storage.get(self.PAR_SELECTED)
            self.PAR_LINE, = self.par_plot.plot(
                [spot_info.xpos, event.xdata],
                [spot_info.ypos, event.ydata],
                "r-", lw=2
            )
            plt.draw()

    def key_press_event(self, event):
        if self.MODE == 1:
            if event.key == "escape":
                self.MODE = 0
                try:
                    self.par_plot.lines.remove(self.PAR_LINE)
                except:
                    pass
                spot_info = self.spot_storage.get(self.PAR_SELECTED)
                spot_info.spot_marker.set_color(spot_info.colour)
                spot_info.spot_line = self.par_plot.add_artist(spot_info.spot_line)
                self.par_plot.lines.append(spot_info.spot_line)

                plt.draw()



def process(f, lineage_num):
    cell_line = np.load(f)
    Connector(cell_line, lineage_num)


def main(wanted=True):
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
        print("No files found")
        return

    for targetfile in targetfiles:
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        if lineage_num in wantedlineages:
            process(targetfile, lineage_num)


if __name__ == "__main__":
    if "-w" in sys.argv:
        wanted = False
    else:
        wanted = True

    main(wanted=wanted)
