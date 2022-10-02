
def duration_to_str(duration):
    milli = duration % 1000
    seconds = (duration // 1000) % 60
    minutes = duration // 60000
    return f'{minutes}min {seconds:2}s {milli:3}ms'


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
