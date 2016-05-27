#!/usr/bin/env python

""" Determine whether cells without ParB foci are also minicells """

import pandas as pd

def main():
    # for all groupings, determine number of ParB spots (total) and at start
    # record num spots, num init spots, num final spots, cell length at birth
    # (or None), cell length at division (or None), cell length at start, cell
    # length at end
    datastruct = [
        "topdir", "subdir",
        "cell_id", "parent_id", "daughter1_id", "daughter2_id",
        "num_spots", "num_init_spots", "num_final_spots",
        "length_birth", "length_division",
        "length_start", "length_end",
    ]

    data = pd.DataFrame(columns=datastruct)
    data = data.append(pd.Series(
        [str(x) for x in range(len(datastruct))],
        index=datastruct, name="test1"
    ))
    data = data.append(pd.Series(
        [str(x) for x in range(len(datastruct), len(datastruct) * 2)],
        index=datastruct, name="test2"
    ))

    print(data)


if __name__ == "__main__":
    main()
