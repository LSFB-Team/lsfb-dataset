import pandas as pd
import numpy as np
from os import path
from typing import Optional


def load_pose_landmarks(root: Optional[str], csv_path: str):
    filepath = path.join(root, csv_path) if root is not None else csv_path
    df = pd.read_csv(filepath, dtype='float32')
    return df


def load_hands_landmarks(root: Optional[str], csv_path: str, hands: str):
    filepath = path.join(root, csv_path) if root is not None else csv_path
    df = pd.read_csv(filepath, dtype='float32')
    if hands == 'right':
        df = df.loc[:, df.columns.str.startswith('RIGHT')]
    elif hands == 'left':
        df = df.loc[:, df.columns.str.startswith('LEFT')]
    return df


def pad_landmarks(landmarks, padding: int):
    if padding > 0:
        landmarks = np.pad(landmarks, ((0, padding), (0, 0)), constant_values=((0, 0), (0, 0)))
    return landmarks
