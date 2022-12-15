import numpy as np
from itertools import groupby
from typing import Optional
from os import path
import pickle


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


def load_segmentation_targets(
        root: str,
        target_vectors_dir: str = 'annotations/vectors',
        target_type: str = 'signs',
):
    if target_type == 'signs':
        vectors_filename = 'binary.pck'
    elif target_type == 'signs_and_transitions':
        vectors_filename = 'binary_with_coarticulation.pck'
    elif target_type == 'activity':
        vectors_filename = 'activity.pck'
    else:
        raise ValueError(f'Unknown target: {target_type}')

    targets_filepath = path.join(root, target_vectors_dir, vectors_filename)
    if not path.isfile(targets_filepath):
        raise FileNotFoundError(f'Target vectors ({target_type}) file not found')

    with open(targets_filepath, 'rb') as file:
        vectors = pickle.load(file)

    return vectors


def load_segmentation_target(
        root: str,
        video_filename: str,
        target_vectors_dir: str = 'annotations/vectors',
        target_type: str = 'signs',
):
    vectors = load_segmentation_targets(
        root=root,
        target_vectors_dir=target_vectors_dir,
        target_type=target_type,
    )

    target = vectors.get(video_filename)
    if target is None:
        raise ValueError(f'Target ({target_type}) not found for video [{video_filename}].')

    return target

