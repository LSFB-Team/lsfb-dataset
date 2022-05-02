import numpy as np


def fill_gaps(vec, max_gap, no_gap=1, fill_with=1):
    last = 0
    gap = False
    new_vec = np.copy(vec)

    for idx in range(len(vec)):
        if vec[idx] == no_gap:
            if gap:
                if idx - last - 1 <= max_gap:
                    new_vec[last+1:idx] = fill_with
                gap = False

            last = idx

        elif not gap:
            gap = True

    return new_vec


def get_gaps(vec, no_gap=0, gap=1):
    last = 0
    gaps = []

    last_state = 0

    for idx in range(len(vec)):
        val = vec[idx]

        if val == no_gap:
            if last_state == gap:
                gaps.append((last+1, idx-1))

            last = idx

        last_state = val

    return gaps
