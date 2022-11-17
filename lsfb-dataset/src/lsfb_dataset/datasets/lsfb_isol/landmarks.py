import pandas as pd
from tqdm import tqdm
from datetime import datetime

from typing import List
from lsfb_dataset.utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from lsfb_dataset.utils.datasets import create_mask
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase


def _load_landmarks(
        root: str,
        videos: pd.DataFrame,
        landmarks: List[str],
        sequence_max_length: int,
        show_progress: bool,
):
    landmark_list = ', '.join(landmarks)
    print(f'Loading features ({landmark_list}) and labels for each isolated sign...')

    features = []
    targets = []

    progress_bar = tqdm(videos.iterrows(), total=videos.shape[0], disable=(not show_progress))
    for index, video in progress_bar:
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

        features.append(pd.concat(data, axis=1).values[:sequence_max_length])
        targets.append(video['class'])

    return features, targets


class LSFBIsolLandmarks(LSFBIsolBase):

    def __init__(self, *args, **kwargs):
        print('-' * 10, 'LSFB ISOL DATASET')
        start_time = datetime.now()

        super(LSFBIsolLandmarks, self).__init__(*args, **kwargs)

        self.features, self.targets = _load_landmarks(
            self.config.root,
            self.config.videos,
            self.config.landmarks,
            self.config.sequence_max_length,
            self.config.show_progress,
        )
        self.labels = self.config.lemmes['lemme']

        print('-' * 10)
        print('loading time:', datetime.now() - start_time)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]
        pad_value = 0

        if self.config.padding:
            pad_value = self.config.sequence_max_length - len(features)
            features = pad_landmarks(features, pad_value)

        if self.config.features_transform is not None:
            features = self.config.features_transform(features)

        if self.config.target_transform is not None:
            target = self.config.target_transform(target)

        if self.config.transform is not None:
            features, target = self.config.transform(features, target)

        if self.config.return_mask:
            mask = create_mask(self.config.sequence_max_length, pad_value, self.config.mask_value)
            if self.config.mask_transform is not None:
                mask = self.config.mask_transform(mask)

            return features, target, mask

        return features, target
