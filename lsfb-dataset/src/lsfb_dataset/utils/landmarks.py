import pandas as pd
import numpy as np
from os import path
from typing import Optional


def load_pose_landmarks(root: Optional[str], csv_path: str) -> pd.DataFrame:
    filepath = path.join(root, csv_path) if root is not None else csv_path
    df = pd.read_csv(filepath, dtype='float32')
    return df


def load_hands_landmarks(root: Optional[str], csv_path: str, hands: str) -> pd.DataFrame:
    filepath = path.join(root, csv_path) if root is not None else csv_path
    df = pd.read_csv(filepath, dtype='float32')
    if hands == 'right':
        df = df.loc[:, df.columns.str.startswith('RIGHT')]
    elif hands == 'left':
        df = df.loc[:, df.columns.str.startswith('LEFT')]
    return df


def load_landmarks(video: pd.Series, root: str, landmarks: list[str]) -> np.ndarray:
    video_landmarks = []
    for landmark in landmarks:
        if landmark == 'pose':
            video_landmarks.append(load_pose_landmarks(root, video['pose']))
        elif landmark == 'hand_left':
            video_landmarks.append(load_hands_landmarks(root, video['hands'], 'left'))
        elif landmark == 'hand_right':
            video_landmarks.append(load_hands_landmarks(root, video['hands'], 'right'))
        else:
            raise ValueError(f'Unknown landmarks: {landmark}.')
    return pd.concat(video_landmarks, axis=1).values


def pad_landmarks(landmarks, padding: int):
    if padding > 0:
        landmarks = np.pad(landmarks, ((0, padding), (0, 0)), constant_values=((0, 0), (0, 0)))
    return landmarks
