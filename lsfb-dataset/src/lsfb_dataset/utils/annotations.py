import pandas as pd
import json
import os


def get_annotations_in_time_range(df_annotations: pd.DataFrame, time_range: (int, int)):
    return df_annotations[
        ((df_annotations['start'] >= time_range[0]) & (df_annotations['start'] <= time_range[1])) |
        ((df_annotations['end'] >= time_range[0]) & (df_annotations['end'] <= time_range[1]))
    ]


def load_glosses(root: str, hand: str):
    assert hand in ('left', 'right')
    with open(os.path.join(root, 'annotations', f'gloss_{hand}.json')) as file:
        return json.load(file)
