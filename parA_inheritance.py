#!/usr/bin/env python

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import seaborn as sns
sns.set_context("paper")
sns.set_style("white")


def get_total_intensity(cell):
    # get mask
    mask_vertices = [cell.mesh[0, 0:2]]
    for coords in cell.mesh[1:-1, 0:2]:
        mask_vertices.append(coords)

    mask_vertices.append(cell.mesh[-1, 0:2])
    for coords in cell.mesh[1:-1, 2:4][::-1]:
        mask_vertices.append(coords)

    mask_vertices.append(cell.mesh[0, 0:2])
    mask_vertices = np.array(mask_vertices)

    x = mask_vertices[:, 0]
    y = mask_vertices[:, 1]
    rows, cols = skimage.draw.polygon(y, x)
    values = cell.parA_img[rows, cols]
    total_intensity = values.sum()

    return total_intensity

def main():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lookup = json.loads(open("ancestry.json").read())
    siblings = {}  # (mother_lin, daughter_lin, daughter_lin
    cell_lines = {}

    lin_nums = range(1, len(lin_files) + 1)
    for l in lin_files:
        c = np.load(l)
        mother_lin = lookup[c[0].id]
        cell_lines[mother_lin] = c
        if c[-1].children:
            siblings[lookup[c[0].id]] = (lookup[c[-1].children[0]], lookup[c[-1].children[1]])

    print(siblings)
    fig = plt.figure()
    p1 = fig.add_subplot(131)
    p2 = fig.add_subplot(132)
    p3 = fig.add_subplot(133)
    for current_num in lin_nums:
        cell_line = cell_lines[current_num]
        t = cell_line[0].t
        i = [get_total_intensity(x) for x in cell_line]
        p1.plot(t, i, lw=2, label=current_num)
        p1.set_title("Intensity")

        i2 = [get_total_intensity(x) / x.length[0][0] for x in cell_line]
        p2.plot(t, i2, lw=2)
        p2.set_title("Length (per px)")

        i3 = [get_total_intensity(x) / x.area[0][0] for x in cell_line]
        p3.plot(t, i3, lw=2)
        p3.set_title("Area (per px^2)")

    p1.legend()
    plt.show()


if __name__ == "__main__":
    main()
