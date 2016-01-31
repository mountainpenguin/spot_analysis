#!/usr/bin/env python

from spot_analysis import SpotTimeLapse
import numpy as np
import json
import datetime


def get_parB_path(cell_line, T):
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
