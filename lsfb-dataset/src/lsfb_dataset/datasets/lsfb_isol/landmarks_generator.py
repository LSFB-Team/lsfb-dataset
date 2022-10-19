from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase

import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from lsfb_dataset.datasets.types import *
from typing import List
from lsfb_dataset.utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from lsfb_dataset.utils.datasets import create_mask
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase



def _load_landmark(
        root: str,
        video: pd.Series,
        landmarks: List[str],
        sequence_max_length: int,
):

    features = []
    targets = []
    data = []

    for lm_type in landmarks:
        if lm_type == 'pose':
            data.append(load_pose_landmarks(root, video['pose']))
        elif lm_type == 'hand_left':
            data.append(load_hands_landmarks(root, video['hands'], 'left'))
        elif lm_type == 'hand_right':
            data.append(load_hands_landmarks(root, video['hands'], 'right'))
        else:
            raise ValueError(f'Unknown landmarks: {lm_type}.')

    features = pd.concat(data, axis=1).values[:sequence_max_length]
    targets = video['class']

    return features, targets


class LSFBIsolLandmarksGenerator(LSFBIsolBase):

    def __init__(self,*args, **kwargs):
        super(LSFBIsolLandmarksGenerator, self).__init__(*args, **kwargs)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        
        video = self.videos.iloc[index]

        features, target = _load_landmark(self.root, video, self.landmarks, self.sequence_max_length)
        pad_value = 0

        if self.padding:
            pad_value = self.sequence_max_length - len(features)
            features = pad_landmarks(features, pad_value)

        if self.transform is not None:
            features = self.transform(features)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_mask:
            mask = create_mask(self.sequence_max_length, pad_value, self.mask_value)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            return features, target, mask

        return features, target
