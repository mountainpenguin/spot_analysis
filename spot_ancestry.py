#!/usr/bin/env python

import glob
import re
import numpy as np
import os
import json
import sys
from lineage_lib import track
import spot_plot
import matplotlib.pyplot as plt
import matplotlib.gridspec
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns
sns.set_style("white")


def getNumByID(cell_id):
    if not os.path.exists("ancestry.json"):
        cells = init()
    else:
        cells = json.loads(open("ancestry.json").read())
    return cells[cell_id]


def init():
    # first_cell_id: lineage_num
    cells = {}
    targetfiles = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    tot = len(targetfiles)
    done = 1
    for targetfile in targetfiles:
        lineage_num = int(re.search("lineage(\d+)\.npy", targetfile).group(1))
        print("Processed {0} of {1} (Lineage {2:02d})".format(done, tot, lineage_num))
        cell_line = np.load(targetfile)
        cells[cell_line[0].id] = lineage_num
        done += 1
    open("ancestry.json", "w").write(json.dumps(cells))
    return cells


def plot_lineages(bonly=False):
    if not os.path.exists("ancestry.json"):
        cells = init()
    else:
        cells = json.loads(open("ancestry.json").read())
    inverse_cells = {cells[k]: k for k in cells}

    lineage = track.Lineage()

    already_plotted = []
    for lineage_num in sorted(inverse_cells.keys()):
        if lineage_num in already_plotted:
            continue

        print("Processing lineage {0}".format(lineage_num))
        cell_line_nums = []
#        cell_line_positions = {}
        cell_lines = {}
        cell_parents = {}
        designations = {
            lineage_num: "",
        }
        current_num = int(lineage_num)
        buff = []
        generations = {}
        max_len = 0

        graph = nx.DiGraph()
        labels = {}

        while True:
            if current_num:
                mother_line = np.load("data/cell_lines/lineage{0:02d}.npy".format(current_num))
                max_len_ = max([x.length[0][0] for x in mother_line])
                if max_len_ > max_len:
                    max_len = float(max_len_)
                cell_lines[current_num] = mother_line
                cell_line_nums.append(current_num)
                daughters = mother_line[-1].children

                gen = 0
                designation = ""
                _temp_num = int(current_num)
                while True:
                    if _temp_num not in cell_parents:
                        break
                    _temp_num = cell_parents[_temp_num]
                    gen += 1
                generations[current_num] = gen

                if mother_line[0].parent:
                    graph.add_edge(mother_line[0].parent, mother_line[-1].id)
                else:
                    graph.add_node(mother_line[-1].id)
                labels[mother_line[-1].id] = current_num

                if daughters:
                    d1 = lineage.frames.cell(daughters[0]).length[0][0]
                    d2 = lineage.frames.cell(daughters[1]).length[0][0]
                    max_len_ = max([max_len_, d1 + d2])
                    if max_len_ > max_len:
                        max_len = float(max_len_)

                    d1n = cells[daughters[0]]
                    d2n = cells[daughters[1]]
                    buff.append(d1n)
                    buff.append(d2n)
                    designations[d1n] = designations[current_num] + "A"
                    designations[d2n] = designations[current_num] + "B"
                    cell_parents[d1n] = current_num
                    cell_parents[d2n] = current_num
                    print("{0} => {1}, {2}".format(current_num, d1n, d2n))
                else:
                    buff.append(None)
                    buff.append(None)
                    print("{0} => None".format(current_num))
            else:
                cell_line_nums.append(None)
            try:
                current_num = buff.pop(0)
            except IndexError:
                break

#        max_gen = max(generations.values())
#
#        last_xpos = None
#        count = 0
#        specs = []
#        gridspec = matplotlib.gridspec.GridSpec(2 ** max_gen, max_gen + 1)
#
#        for num in cell_line_nums:
#            if not num:
#                count += 1
#                continue
#            # get generation
#            xpos = generations[num]
#            if xpos == last_xpos:
#                count += 1
#            else:
#                last_xpos = int(xpos)
#                count = 0
#            yinterval = 2 ** (max_gen - xpos)
#            ypos = count * yinterval
#            specs.append(gridspec[ypos, xpos])
#
#        fig = plt.figure(
#            figsize=(
#                (20 * (max_gen + 1)) / 6,
#                (5 * (2 ** max_gen)) / 2
#            )
#        )
#
#        i = 0
#        for cell_line_num in cell_line_nums:
#            if not cell_line_num:
#                continue
#
#            cell_line = cell_lines[cell_line_num]
#            print("Plotting lineage {0}".format(cell_line_num))
#
#            ax = fig.add_subplot(specs[i])
#            spot_plot._despine(ax)
#            ax.set_title("Lineage {0}".format(cell_line_num))
#            spot_plot.plot_graphs_parB_only(cell_line, cell_line_num, ax_parB=ax, save=False)
#            ylim = (max_len + 5) / 2
#            ax.set_ylim([ylim, -ylim])
#
#            already_plotted.append(cell_line_num)
#            i += 1
#

        num_rows = int(round(np.sqrt(len(cell_lines) + 1)))
        if num_rows ** 2 < len(cell_lines) + 1:
            gridspec = matplotlib.gridspec.GridSpec(num_rows + 1, num_rows)
        else:
            gridspec = matplotlib.gridspec.GridSpec(num_rows, num_rows)
        fig = plt.figure(
            figsize=(
                (10 * num_rows) / 3,
                (5 * num_rows) / 2
            )
        )
        fig.patch.set_alpha(0)

        i = 0
        for cell_line_num in cell_line_nums:
            if not cell_line_num:
                continue
            designation = designations[cell_line_num]
            title = "{0} ({1})".format(designation, cell_line_num)

            cell_line = cell_lines[cell_line_num]
            print("Plotting lineage {0}".format(cell_line_num))

            sp = matplotlib.gridspec.SubplotSpec(gridspec, i)
            ax = fig.add_subplot(sp)
            ax.set_title(title)
            spot_plot._despine(ax)
            if bonly:
                spot_plot.plot_graphs_parB_only(cell_line, cell_line_num, ax_parB=ax, save=False)
            else:
                spot_plot.plot_graphs(cell_line, cell_line_num, parA_heatmap=ax, save=False, num_plots=1)

            ylim = (max_len + 5) / 2
            ax.set_ylim([-ylim, ylim])

            already_plotted.append(cell_line_num)
            i += 1

        # add network plot
        sp = matplotlib.gridspec.SubplotSpec(gridspec, i)
        ax = fig.add_subplot(sp)
        ax.set_title("Tree")
        spot_plot._despine(ax)
        pos = graphviz_layout(graph, prog="dot")
        nx.draw_networkx_labels(
            graph, pos, labels=labels,
            font_size=10,
            font_weight="bold",
        )
        nx.draw(
            graph, pos, arrows=False,
            node_color=sns.color_palette()[0],
            width=2,
        )

        plt.tight_layout()

        os.makedirs("data/full_lineages", exist_ok=True)
        plt.savefig("data/full_lineages/lineage{0:02d}.pdf".format(lineage_num))

        print("Done with tree starting from {0}".format(lineage_num))


def main():
    if "plot" in sys.argv:
        if "-b" in sys.argv:
            plot_lineages(bonly=True)
        else:
            plot_lineages()
    else:
        if not os.path.exists("ancestry.json"):
            init()
            print("Ancestries determined")
        else:
            print("Ancestries already determined")

if __name__ == "__main__":
    main()
