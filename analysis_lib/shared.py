#!/usr/bin/env python

import numpy as np
import json
import datetime
import hashlib
import random
import string
import os
import matplotlib.pyplot as plt
import scipy.stats


class TraceConnect(object):
    def __init__(self, t, pos, intensity, length, ID):
        self.id = hashlib.sha1("".join([
            random.choice(
                string.ascii_letters + string.digits
            ) for x in range(40)
        ]).encode("utf8")).hexdigest()
        self.spot_ids = [ID]
        self.timing = [t]
        if pos < 0:
            self.POLE = -1
        else:
            self.POLE = 1
        self.position = [pos]
        self.intensity = [intensity]
        self.length = [length]
        self.split_parent = None
        self.split_children = None

    def spots(self, adjust=False):
        data = []
        for x in range(len(self.timing)):
            data.append((
                self.timing[x],
                self.position[x],
                self.intensity[x],
                self.spot_ids[x].encode("ascii")
            ))
        return np.array(data, dtype=[
            ("timing", "i4"),
            ("position", "f4"),
            ("intensity", "f4"),
            ("id", "S20")
        ])

    def len(self):
        return np.array(self.length)

    def append(self, t, pos, intensity, length, ID):
        self.timing.append(t)
        self.position.append(pos)
        self.intensity.append(intensity)
        self.length.append(length)
        self.spot_ids.append(ID)

    def __len__(self):
        return len(self.length)

    def add_split_parent(self, ID):
        self.split_parent = ID

    def add_split_children(self, child1, child2):
        self.split_children = [child1, child2]


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
        data = []
        for x in range(len(self.timing)):
            data.append((
                self.timing[x],
                adjust and self.POLE * self.positions[x] or self.positions[x],
                self.intensity[x]
            ))
        return np.array(data, dtype=[
            ("timing", "i4"),
            ("position", "f4"),
            ("intensity", "f4")
        ])

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


def get_parA_path(cell_line, T):
    start = cell_line[0].frame - 1
    end = cell_line[-1].frame - 1
    t = np.array(T[start:end + 1])
    i = 0
    for tp in cell_line:
        pos, inten, celllen = tp.ParA
        if i == 0:
            spots_ParA = SpotTimeLapse(
                t[i],
                pos - (celllen / 2),
                inten,
                celllen,
            )
        else:
            spots_ParA.append(
                t[i],
                pos - (celllen / 2),
                inten,
                celllen,
            )
        i += 1
    return [spots_ParA]


def get_parB_path(cell_line, T, lineage_num, force=False):
    if not force:
        # check for pre-existing file
        lineage_num = int(lineage_num)
        target = "data/spot_data/lineage{0:02d}.npy".format(lineage_num)
        if os.path.exists(target):
            spots_ParB = np.load(target)
            return sorted(spots_ParB, key=lambda x: x.timing[0])

    start = cell_line[0].frame - 1
    end = cell_line[-1].frame - 1
    t = np.array(T[start:end + 1])
    ParB_data = [x.ParB for x in cell_line]
    spots_ParB = []
    i = 0
    for tp in ParB_data:
        if i > 0:
            tp_clone = list(tp)
            options = list(spots_ParB)
            TRAVEL_THRESHOLD = 5.1
            for opt in options:
                last_added = opt.last()
                if last_added[0] < t[i] - 15 * 4:
                    # more than 3 timepoints ago
                    continue

                distances = []
                if len(tp_clone) == 0:
                    continue

                for tp_idx in range(len(tp_clone)):
                    pos, inten, celllen = tp_clone.pop()
                    distances.append((
                        tp_idx, pos, inten, celllen,
                        np.sqrt(
                            ((pos - (celllen / 2)) - last_added[1]) ** 2
                        )
                    ))
                d = np.array(distances)
                min_idx, min_pos, min_inten, min_celllen, min_dist = d[
                    np.argmin(d[:, 4])
                ]
                dd_idx = 0
                for dd in distances:
                    if dd_idx == min_idx and min_dist < TRAVEL_THRESHOLD:
                        opt.append(
                            t[i],
                            min_pos - (min_celllen / 2),
                            min_inten,
                            min_celllen,
                        )
                    else:
                        tp_idx, pos, inten, celllen, dist = dd
                        tp_clone.append((
                            pos,
                            inten,
                            celllen,
                        ))
                    dd_idx += 1

            # start new spot timelapses for unassigned foci
            for pos, inten, celllen in tp_clone:
                spots_ParB.append(SpotTimeLapse(
                    t[i],
                    pos - (celllen / 2),
                    inten,
                    celllen,
                ))

        else:
            # start new spot timelapses
            for pos, inten, celllen in tp:
                spots_ParB.append(SpotTimeLapse(
                    t[i],
                    pos - (celllen / 2),
                    inten,
                    celllen,
                ))

        i += 1

    return sorted(spots_ParB, key=lambda x: x.timing[0])


def get_timings(t0=False):
    timing_data = json.loads(open("timings.json").read())
    timings = timing_data["timings"]
    T = []
    if "start" in timing_data:
        t0 = _gettimestamp(*timing_data["start"])
    else:
        t0 = _gettimestamp(*timings[0])
    for d1, t1, frames in timings:
        sm = _timediff(d1, t1, t0)
        for _ in range(frames):
            T.append(sm)
            sm += timing_data["pass_delay"]
    return T


def _gettimestamp(day, time, *args):
    return datetime.datetime.strptime(
        "{0} {1}".format(day, time),
        "%d.%m.%y %H:%M"
    )


def _timediff(day, time, t0):
    t1 = _gettimestamp(day, time)
    td = t1 - t0
    s = td.days * 24 * 60 * 60
    s += td.seconds
    m = s // 60
    return m


def get_growth_rate(lin, PX=0.12254):
    """ Exponential method of determining growth rate """
    if len(lin) < 3:
        return None
    timings = (lin[0].t - lin[0].t[0]) / 60
    lengths = [x.length[0][0] * PX for x in lin]

    logLength = np.log(lengths)
    return np.polyfit(timings, logLength, 1)[0]


def get_elongation_rate(lin, PX=0.12254, discard=False):
    """ Linear method of determining growth rate """
    if len(lin) < 3:
        return None
    timings = lin[0].t / 60
    lengths = [x.length[0][0] * PX for x in lin]
    elongation_rate = np.polyfit(timings, lengths, 1)[0]
    if discard and elongation_rate < 0:
        return 0
    return elongation_rate


def add_stats(data, xkey, ykey, ax=None, m=True, r2=True, n=True):
    if not ax:
        ax = plt.gca()
    plt.sca(ax)
    dataset = data[[xkey, ykey]].dropna()
    x = dataset[xkey]
    y = dataset[ykey]

    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x, y)
    if m:
        plt.plot(x[0], y[0], color="none", alpha=1, label=r"$m = {0:.5f}$".format(slope))
    if r2:
        plt.plot(x[0], y[0], color="none", alpha=1, label=r"$r^2 = {0:.5f}$".format(rvalue ** 2))
    if n:
        plt.plot(x[0], y[0], color="none", alpha=1, label=r"$n = {0}$".format(len(dataset)))
    plt.legend()
