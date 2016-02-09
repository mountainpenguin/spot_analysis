#!/usr/bin/env python

import numpy as np
import json
import datetime
import hashlib
import random
import string
import os


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

    def spots(self, adjust=False):
        return np.array([
            self.timing,
            self.position,
            self.intensity
        ]).T

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
        if adjust:
            return np.array([
                self.timing,
                self.POLE * np.array(self.positions),
                self.intensity
            ]).T
        else:
            return np.array([
                self.timing,
                self.positions,
                self.intensity
            ]).T

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
            return spots_ParB

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

    return spots_ParB


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
