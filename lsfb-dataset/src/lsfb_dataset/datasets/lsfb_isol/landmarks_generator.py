import pandas as pd

from lsfb_dataset.utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from lsfb_dataset.utils.datasets import create_mask
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase


class LSFBIsolLandmarksGenerator(LSFBIsolBase):

    def __init__(self, *args, **kwargs):
        super(LSFBIsolLandmarksGenerator, self).__init__(*args, **kwargs)
        self.labels = self.config.lemmes['lemme']

    def __len__(self):
        return len(self.config.videos)

    def __getitem__(self, index):

        video = self.config.videos.iloc[index]

        features, target = self._load_landmark(video)
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

    def _load_landmark(self, video):
        data = []

        landmarks = self.config.landmarks
        root = self.config.root

        for lm_type in landmarks:
            if lm_type == 'pose':
                data.append(load_pose_landmarks(root, video['pose']))
            elif lm_type == 'hand_left':
                data.append(load_hands_landmarks(root, video['hands'], 'left'))
            elif lm_type == 'hand_right':
                data.append(load_hands_landmarks(root, video['hands'], 'right'))
            else:
                raise ValueError(f'Unknown landmarks: {lm_type}.')

        features = pd.concat(data, axis=1).values[:self.config.sequence_max_length]
        targets = video['class']

        return features, targets
