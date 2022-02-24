import pandas as pd
import numpy as np
from typing import Optional


def get_annotations_in_time_range(df_annotations: pd.DataFrame, time_range: (int, int)):
    return df_annotations[
        ((df_annotations['start'] >= time_range[0]) & (df_annotations['start'] <= time_range[1])) |
        ((df_annotations['end'] >= time_range[0]) & (df_annotations['end'] <= time_range[1]))
    ]


def get_annotations_durations(annot_vec: list[bool], value=True, fps=50.0):
    durations = []
    current_duration = 0
    ms_per_frame = 1000/fps
    for x in annot_vec:
        if x == value:
            current_duration = current_duration + ms_per_frame
        elif current_duration > 0:
            durations.append(current_duration)
            current_duration = 0
    return durations


def annotations_to_vec(annotations_right: Optional[pd.DataFrame], annotations_left: Optional[pd.DataFrame],
                       frames_nb: int, fps=50.0):
    vec = np.full(frames_nb, False, dtype=bool)
    frames_per_ms = fps/1000

    if annotations_right is not None:
        for _, annot in annotations_right.iterrows():
            begin_frame = int(annot['start']*frames_per_ms)
            end_frame = int(annot['end']*frames_per_ms)
            vec[begin_frame:end_frame] = True

    if annotations_left is not None:
        for _, annot in annotations_left.iterrows():
            begin_frame = int(annot['start']*frames_per_ms)
            end_frame = int(annot['end']*frames_per_ms)
            vec[begin_frame:end_frame] = True

    return vec


def create_coerc_vec(annot_vec, max_coerc_size=50):
    last_word = -1
    coerc_vec = np.zeros(len(annot_vec), dtype='ubyte')
    for idx in range(len(annot_vec)):
        if annot_vec[idx]:
            if last_word >= 0:
                coerc_vec[max(0, idx-last_word):idx] = 2

            coerc_vec[idx] = 1
            last_word = 0
        else:
            if last_word >= 0:
                last_word += 1
                if last_word >= max_coerc_size:
                    last_word = -1

    return coerc_vec
