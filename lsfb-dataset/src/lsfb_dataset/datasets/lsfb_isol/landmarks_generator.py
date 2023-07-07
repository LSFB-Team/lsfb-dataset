import pandas as pd

from lsfb_dataset.utils.landmarks import (
    load_pose_landmarks,
    load_hands_landmarks,
    pad_landmarks,
)
from lsfb_dataset.utils.datasets import create_mask
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase
import numpy as np


class LSFBIsolLandmarksGenerator(LSFBIsolBase):
    def __init__(self, *args, **kwargs):
        super(LSFBIsolLandmarksGenerator, self).__init__(*args, **kwargs)

    def __len__(self):
        return len(self.config.instances)

    def __getitem__(self, index):
        instance = self.config.instances.iloc[index]

        features, target = self._load_landmark(instance)
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
            mask = create_mask(
                self.config.sequence_max_length, pad_value, self.config.mask_value
            )
            if self.config.mask_transform is not None:
                mask = self.config.mask_transform(mask)

            return features, target, mask

        return features, target

    def _load_landmark(self, instance):
        features = {}

        landmarks = self.config.landmarks
        root = self.config.root

        for lm_type in landmarks:
            features[lm_type] = np.load(f"{root}/poses/{lm_type}/{instance['id']}.npy")

        target = instance["sign"]

        return features, target
