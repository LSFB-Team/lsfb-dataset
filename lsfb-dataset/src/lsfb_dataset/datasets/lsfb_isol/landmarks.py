import pandas as pd
from tqdm import tqdm
from datetime import datetime

from typing import List
from lsfb_dataset.utils.landmarks import (
    load_pose_landmarks,
    load_hands_landmarks,
    pad_landmarks,
)
from lsfb_dataset.utils.datasets import create_mask
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase
import numpy as np
import os


def _load_landmarks(
    root: str,
    instances: pd.DataFrame,
    landmarks: List[str],
    sequence_max_length: int,
    show_progress: bool,
):
    landmark_list = ", ".join(landmarks)
    print(f"Loading features ({landmark_list}) and labels for each isolated sign...")

    features = []
    targets = []

    progress_bar = tqdm(
        instances.iterrows(), total=instances.shape[0], disable=(not show_progress)
    )

    features = {}

    for _, instance in progress_bar:
        for lm_type in landmarks:
            if lm_type not in features:
                features[lm_type] = []

            data = np.load(f"{root}/poses/{lm_type}/{instance['id']}.npy")
            features[lm_type].append(data)

        targets.append(instance["sign"])

    return features, targets


class LSFBIsolLandmarks(LSFBIsolBase):
    def __init__(self, *args, **kwargs):
        print("-" * 10, "LSFB ISOL DATASET")
        start_time = datetime.now()

        super(LSFBIsolLandmarks, self).__init__(*args, **kwargs)

        self.features, self.targets = _load_landmarks(
            self.config.root,
            self.config.instances,
            self.config.landmarks,
            self.config.sequence_max_length,
            self.config.show_progress,
        )

        print("-" * 10)
        print("loading time:", datetime.now() - start_time)

    def __len__(self):
        return len(self.config.instances)

    def __getitem__(self, index):
        features = {}

        for key in features:
            features[key] = self.features[key][index]

        target = self.targets[index]
        pad_value = 0

        # TODO refactor the transforms

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
