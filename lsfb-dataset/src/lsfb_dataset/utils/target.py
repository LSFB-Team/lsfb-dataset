import numpy as np


def pad_target(target, padding: int):
    if padding > 0:
        target = np.pad(target, (0, padding), constant_values=(0, 0))
    return target
