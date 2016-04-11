#!/usr/bin/env python

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import seaborn as sns
sns.set_context("paper")
sns.set_style("white")
import os
# import pandas as pd


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
    xlim, ylim = cell.parA_img.shape
    rows[rows >= xlim] = xlim - 1
    cols[cols >= ylim] = ylim - 1
    values = cell.parA_img[rows, cols]
    total_intensity = values.sum()

    return total_intensity


def process():
    lin_files = sorted(glob.glob("data/cell_lines/lineage*.npy"))
    lookup = json.loads(open("ancestry.json").read())
    siblings = {}  # (mother_lin, daughter_lin, daughter_lin
    cell_lines = {}

    for l in lin_files:
        c = np.load(l)
        mother_lin = lookup[c[0].id]
        cell_lines[mother_lin] = c
        if c[-1].children:
            siblings[lookup[c[0].id]] = (lookup[c[-1].children[0]], lookup[c[-1].children[1]])

    for parent_num in sorted(siblings.keys()):
        child1_num, child2_num = siblings[parent_num]
#        parent = cell_lines[parent_num][-1]
        child1 = cell_lines[child1_num][0]
        child2 = cell_lines[child2_num][0]

        # make child1 the smaller cell
        if child1.length < child2.length:
            child2_num, child1_num = siblings[parent_num]
            child1 = cell_lines[child1_num][0]
            child2 = cell_lines[child2_num][0]

        c1_inten = get_total_intensity(child1)
        c2_inten = get_total_intensity(child2)

        c_ratio = c1_inten / c2_inten  # ratio of intensity between children
        l_ratio = (child1.length / child2.length)[0][0]  # ratio of child lengths
#        a_ratio = (child1.area / child2.area)[0][0]  # ratio of child areas
#        dl_ratio = c_ratio / l_ratio  # prop. to length
#        da_ratio = c_ratio / a_ratio  # prop. to area

        plt.plot(l_ratio, c_ratio, marker="o")


def main():
    plt.figure()

    # iterate through all folders
    original_path = os.getcwd()
    dirs = filter(lambda x: os.path.isdir(x), os.listdir())
    data = {}
    for d in dirs:
        subdirs = filter(lambda y: os.path.isdir(os.path.join(d, y)), os.listdir(d))
        for subdir in subdirs:
            exists = ["mt", "ancestry.json", "lineages.json", "data/cell_lines/lineage01.npy"]
            if sum([os.path.exists(os.path.join(d, subdir, z)) for z in exists]) == len(exists):
                os.chdir(os.path.join(d, subdir))
                print("< {0}".format(os.path.join(d, subdir)))
                out = process()
                if out:
                    data[os.path.join(d, subdir)] = out
                os.chdir(original_path)

    sns.despine()
    plt.xlabel("ratio in length")
    plt.ylabel("ratio in intensity")
    xlim = plt.xlim()
    plt.plot(xlim, xlim, "k--")
    plt.savefig("ParA_inheritance/ratios.pdf")
    plt.show()

if __name__ == "__main__":
    main()
