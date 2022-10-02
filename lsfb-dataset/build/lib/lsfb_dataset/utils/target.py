import numpy as np
from itertools import groupby
from typing import Optional


def pad_target(target, padding: int):
    if padding > 0:
        target = np.pad(target, (0, padding), constant_values=(0, 0))
    return target


def combine_binary_vectors(vec1, vec2):
    return np.logical_or(vec1 == 1, vec2 == 1).astype('uint8')


def combine_binary_vectors_with_coarticulation(vec1, vec2):
    vec = 2 * np.logical_or(vec1 == 2, vec2 == 2).astype('uint8')
    vec[vec1 == 1] = 1
    vec[vec2 == 1] = 1
    return vec


def get_segments(segmentation: np.ndarray, filter_value: Optional[int] = None):
    segments = np.array([[key, 0, len(list(val))] for key, val in groupby(segmentation)])
    segments[1:, 1] = np.cumsum(segments[:-1, 2], axis=0)
    segments[:, 2] += segments[:, 1] - 1

    if filter_value is not None:
        segments = segments[segments[:, 0] == filter_value][:, 1:]

    return segments
