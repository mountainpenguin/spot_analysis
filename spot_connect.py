#!/usr/bin/env python

import sys
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import random
import string
import seaborn as sns
sns.set_style("white")

from analysis_lib import shared


class SpotPicker(object):
    def __init__(self, ID, spot_marker, spot_line, xpos, ypos, colour, spot_parent, spot_child, intensity, cell_length, split_parent=None, split_children=None):
        self.ID = ID
        self.spot_marker = spot_marker
        self.spot_line = spot_line
        self.xpos = xpos
        self.ypos = ypos
        self.colour = colour
        self.spot_parent = spot_parent
        self.spot_child = spot_child
        self.intensity = intensity
        self.cell_length = cell_length
        self.split_parent = split_parent
        self.split_children = split_children


class SpotStorage(object):
    def __init__(self):
        self.spots = {}

    def add(self, spot_marker, spot_line, xpos, ypos, colour, spot_parent, spot_child, intensity, cell_length, ID=None, split_parent=None, split_children=None):
        if not ID:
            ID = "".join([random.choice(string.ascii_letters) for x in range(20)])

        if spot_parent:
            spot_parent = self.get(spot_parent)
        if spot_child:
            spot_child = self.get(spot_child)

        self.spots[ID] = SpotPicker(
            ID, spot_marker, spot_line, xpos, ypos, colour,
            spot_parent, spot_child, intensity, cell_length,
            split_parent, split_children,
        )
        return ID

    def get(self, ID):
        return self.spots[ID]

    def get_progenitors(self):
        ret = []
        for ID, data in self.spots.items():
            if not data.spot_parent:
                ret.append(data)
        return ret

    def get_lineage_from(self, ID):
        spot1 = self.get(ID)
        lin = [spot1]
        while spot1.spot_child:
            lin.append(spot1.spot_child)
            spot1 = spot1.spot_child
        return lin

    def set(self, ID, spot_info):
        self.spots[ID] = spot_info

    def set_child(self, ID, spot_child_id):
        if spot_child_id:
            spot_child = self.get(spot_child_id)
        else:
            spot_child = None
        self.spots[ID].spot_child = spot_child

    def set_parent(self, ID, spot_parent_id):
        if spot_parent_id:
            spot_parent = self.get(spot_parent_id)
        else:
            spot_parent = None
        self.spots[ID].spot_parent = spot_parent

    def split_parent(self, ID, split_parent_id):
        if split_parent_id:
            split_parent = self.get(split_parent_id)
        else:
            split_parent = None
        self.spots[ID].split_parent = split_parent

    def split_children(self, ID, split_child1, split_child2=None):
        if not split_child1:
            self.spots[ID].split_children = None
        else:
            self.spots[ID].split_children = [
                self.get(split_child1),
                self.get(split_child2)
            ]


class Connector(object):
    def __init__(self, cell_line, lineage_num):
        self.cell_line = cell_line
        self.lineage_num = lineage_num
        self.T = cell_line[0].T
        self.t = cell_line[0].t
        self.L = np.array([x.length[0][0] for x in cell_line])
        self.spots = shared.get_parB_path(self.cell_line, self.T, self.lineage_num)
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

        self.par_plot = self.fig.add_subplot(111)
        self.par_plot.spines["right"].set_color("none")
        self.par_plot.spines["top"].set_color("none")
        self.par_plot.xaxis.set_ticks_position("bottom")
        self.par_plot.yaxis.set_ticks_position("left")

        self.bold_font = matplotlib.font_manager.FontProperties()
        self.bold_font.set_weight("bold")

        self.mode_default()

    def mode_default(self):
        self.MODE = 0
        self.redraw_par()
        self.par_plot.set_title(
            "Lineage {0}: Default Mode".format(self.lineage_num),
            fontproperties=self.bold_font
        )
        plt.draw()

    def mode_connect(self):
        self.par_plot.set_title(
            "Lineage {0}: Connect Mode".format(self.lineage_num),
            fontproperties=self.bold_font
        )
        self.MODE = 1
        plt.draw()

    def mode_split(self):
        self.par_plot.set_title(
            "Lineage {0}: Split Mode".format(self.lineage_num),
            fontproperties=self.bold_font
        )
        self.MODE = 2
        plt.draw()

    def plot_parB(self):
        # plot cell lengths
        self.par_plot.plot(
            self.t, self.L / 2,
            "k-", lw=2
        )
        self.par_plot.plot(
            self.t, -(self.L / 2),
            "k-", lw=2
        )

        colourwheel = sns.color_palette(n_colors=len(self.spots))
        spotnum = 1
        defer_commands = []
        for x in self.spots:
            colour = colourwheel[spotnum - 1]
            s = x.spots(False)
            index = 0
            spot_ids = []
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

                if index == 0:
                    spot_parent = None
                else:
                    spot_parent = spot_ids[index - 1]

                spot_child = None
                if len(spot) == 4:
                    spot_marker_id = spot[3].decode("utf-8")
                else:
                    spot_marker_id = None
                spot_marker_id = self.spot_storage.add(
                    spot_marker,
                    spot_line,
                    spot[0],
                    spot[1],
                    colour,
                    spot_parent,
                    spot_child,
                    spot[2],
                    x.len()[index],
                    ID=spot_marker_id,
                )
                spot_marker.marker_id = spot_marker_id
                spot_ids.append(spot_marker_id)

                # assign child to parent
                if spot_parent:
                    self.spot_storage.set_child(spot_parent, spot_marker_id)

                index += 1

            # deal with splits
            if hasattr(x, "split_parent") and x.split_parent:
                split_parent = self.spot_storage.get(x.split_parent)
                # first spot in lineage
                first_spot = self.spot_storage.get(spot_ids[0])
                self.spot_storage.split_parent(first_spot.ID, x.split_parent)

                self.par_plot.plot(
                    [split_parent.xpos, first_spot.xpos],
                    [split_parent.ypos, first_spot.ypos],
                    color=first_spot.colour
                )

            if hasattr(x, "split_children") and x.split_children:
                last_child = spot_ids[-1]
                defer_commands.append((
                    self.spot_storage.split_children,
                    (last_child, x.split_children[0], x.split_children[1])
                ))

            spotnum += 1

        for cmd, args in defer_commands:
            cmd(*args)
        self.par_plot.patch.set_alpha(0)

    def pick_event(self, event):
        if self.MODE == 0:
            if event.mouseevent.button == 1:
                event.artist.set_color("r")
                spot_info = self.spot_storage.get(event.artist.marker_id)
                if spot_info.spot_line:
                    self.par_plot.lines.remove(spot_info.spot_line)
                self.PAR_SELECTED = event.artist.marker_id
                self.mode_connect()
            elif event.mouseevent.button == 3:
                event.artist.set_color("g")
                spot_info = self.spot_storage.get(event.artist.marker_id)
                self.PAR_SELECTED = event.artist.marker_id
                self.mode_split()
        elif self.MODE == 1:
            # connect to new spot
            # check forward in time!
            curr_spot = self.spot_storage.get(self.PAR_SELECTED)
            new_child = self.spot_storage.get(event.artist.marker_id)
            if new_child.xpos <= curr_spot.xpos:
                return

            if new_child.spot_parent:
                old_parent = new_child.spot_parent
                try:
                    self.par_plot.lines.remove(old_parent.spot_line)
                except:
                    pass
                old_parent.spot_line = None
                old_parent_child = old_parent.spot_child
                self.spot_storage.set_parent(old_parent_child.ID, None)
                self.spot_storage.set_child(old_parent.ID, None)

            if curr_spot.spot_child:
                old_child = curr_spot.spot_child
                self.spot_storage.set_parent(old_child.ID, None)

            self.spot_storage.set_child(curr_spot.ID, new_child.ID)
            self.spot_storage.set_parent(new_child.ID, curr_spot.ID)

            try:
                self.par_plot.lines.remove(self.PAR_LINE)
            except:
                pass

            self.mode_default()

        elif self.MODE == 2:
            curr_spot = self.spot_storage.get(self.PAR_SELECTED)
            if not curr_spot.spot_child:
                return

            new_child = self.spot_storage.get(event.artist.marker_id)
            if new_child.xpos <= curr_spot.xpos:
                return

            old_child = curr_spot.spot_child

            # curr_spot old_child changes:
            # add split_parent: curr_spot
            # parent to None
            self.spot_storage.split_parent(
                old_child.ID,
                curr_spot.ID
            )
            self.spot_storage.set_parent(
                old_child.ID,
                None
            )

            # curr_spot changes:
            # add split_children to curr_spot
            #   (current spot_child and new_child)
            # spot_child to None
            self.spot_storage.split_children(
                curr_spot.ID,
                old_child.ID,
                new_child.ID
            )
            self.spot_storage.set_child(
                curr_spot.ID,
                None
            )

            # new_child parent
            # spot_child to None
            new_child_parent = new_child.spot_parent
            if new_child_parent:
                self.spot_storage.set_child(
                    new_child_parent.ID,
                    None
                )

            # new_child
            # add split_parent: curr_spot
            # parent to None
            self.spot_storage.split_parent(
                new_child.ID,
                curr_spot.ID
            )
            self.spot_storage.set_parent(
                new_child.ID,
                None
            )

            self.mode_default()

    def motion_notify_event(self, event):
        if self.MODE == 1 or self.MODE == 2:
            try:
                self.par_plot.lines.remove(self.PAR_LINE)
            except:
                pass

            spot_info = self.spot_storage.get(self.PAR_SELECTED)
            if self.MODE == 1:
                linestyle = "r-"
            elif self.MODE == 2:
                linestyle = "g-"
            self.PAR_LINE, = self.par_plot.plot(
                [spot_info.xpos, event.xdata],
                [spot_info.ypos, event.ydata],
                linestyle,
                lw=2
            )
            plt.draw()

    def key_press_event(self, event):
        if (self.MODE == 1 or self.MODE == 2) and event.key == "escape":
            try:
                self.par_plot.lines.remove(self.PAR_LINE)
            except:
                pass
            spot_info = self.spot_storage.get(self.PAR_SELECTED)
            spot_info.spot_marker.set_color(spot_info.colour)
            if self.MODE == 1 and spot_info.spot_line:
                spot_info.spot_line = self.par_plot.add_line(spot_info.spot_line)

            self.mode_default()
            return
        elif self.MODE == 1 and event.key == "enter":
            # disconnect child
            # remove PAR_LINE
            try:
                self.par_plot.lines.remove(self.PAR_LINE)
            except:
                pass
            spot_info = self.spot_storage.get(self.PAR_SELECTED)
            spot_info.spot_marker.set_color(spot_info.colour)
            spot_info.spot_line = None
            self.mode_default()
            return
        elif self.MODE == 0 and event.key == "enter":
            # save results
            self.update_parB()
            plt.close()
        elif self.MODE == 0 and event.key == "escape":
            plt.close()

    def redraw_par(self):
        self.par_plot.clear()

        # plot cell lengths
        self.par_plot.plot(
            self.t, self.L / 2,
            "k-", lw=2
        )
        self.par_plot.plot(
            self.t, -(self.L / 2),
            "k-", lw=2
        )

        self.par_plot.patch.set_alpha(0)
        progenitors = list(self.spot_storage.get_progenitors())
        colourwheel = sns.color_palette("husl", len(progenitors))
        spotnum = 1
        additions = []
        child_updates = {}
        split_parent_updates = {}
        split_child_updates = {}

        for spot in progenitors:
            colour = colourwheel[spotnum - 1]
            spot_lineage = self.spot_storage.get_lineage_from(spot.ID)
            index = 0
            for s in spot_lineage:
                spot_marker, = self.par_plot.plot(
                    s.xpos, s.ypos,
                    marker=".", mec="k",
                    ms=10, color=colour
                )
                spot_marker.marker_id = s.ID
                spot_marker.set_picker(5)
                spot_line = None
                if s.spot_child:
                    spot_line, = self.par_plot.plot(
                        [s.xpos, s.spot_child.xpos],
                        [s.ypos, s.spot_child.ypos],
                        color=colour
                    )
                if s.spot_parent:
                    child_updates[s.spot_parent.ID] = s.ID

                spot_p = s.spot_parent and s.spot_parent.ID or None

                data = {
                    "spot_marker": spot_marker,
                    "spot_line": spot_line,
                    "xpos": s.xpos,
                    "ypos": s.ypos,
                    "colour": colour,
                    "spot_parent": spot_p,
                    "spot_child": None,
                    "intensity": s.intensity,
                    "cell_length": s.cell_length,
                    "ID": s.ID
                }
                additions.append(data)

                if hasattr(s, "split_parent") and s.split_parent:
                    split_parent_updates[s.ID] = s.split_parent.ID

                if hasattr(s, "split_children") and s.split_children:
                    c1, c2 = s.split_children
                    split_child_updates[s.ID] = (c1.ID, c2.ID)

                index += 1
            spotnum += 1

        for add_ in additions:
            self.spot_storage.add(**add_)

        for p, c in child_updates.items():
            self.spot_storage.set_child(p, c)

        for p1, p2 in split_parent_updates.items():
            self.spot_storage.split_parent(p1, p2)
            # draw line
            parent1 = self.spot_storage.get(p1)
            parent2 = self.spot_storage.get(p2)
            self.par_plot.plot(
                [parent2.xpos, parent1.xpos],
                [parent2.ypos, parent1.ypos],
                color=parent1.colour
            )

        for p, (c1, c2) in split_child_updates.items():
            self.spot_storage.split_children(p, c1, c2)

    def update_parB(self):
        spots = []
        for progenitor in self.spot_storage.get_progenitors():
            tl = shared.TraceConnect(
                t=progenitor.xpos,
                pos=progenitor.ypos,
                intensity=progenitor.intensity,
                length=progenitor.cell_length,
                ID=progenitor.ID,
            )
            if progenitor.split_parent:
                tl.add_split_parent(progenitor.split_parent.ID)
            while progenitor.spot_child:
                progenitor = progenitor.spot_child
                tl.append(
                    t=progenitor.xpos,
                    pos=progenitor.ypos,
                    intensity=progenitor.intensity,
                    length=progenitor.cell_length,
                    ID=progenitor.ID,
                )
            if progenitor.split_children:
                child1, child2 = progenitor.split_children
                tl.add_split_children(child1.ID, child2.ID)
            spots.append(tl)

        os.makedirs("data/spot_data", exist_ok=True)
        np.save(
            "data/spot_data/lineage{0:02d}.npy".format(self.lineage_num),
            spots
        )


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
