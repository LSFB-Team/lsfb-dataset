import gc

import numpy as np
from tqdm import tqdm

from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase


class LSFBIsolLandmarks(LSFBIsolBase):
    def __init__(self, config: LSFBIsolConfig):
        super().__init__(config)
        self.features = self._load_features()

    def __getitem__(self, index):
        instance_id = self.instances[index]
        features = self.features[instance_id]

        target = self.targets[instance_id]
        target = self.label_to_index[target]

        if self.config.transform is not None:
            features = self.config.transform(features)

        return features, target

    def _load_features(self):
        pose_folder = "poses_raw" if self.config.use_raw else "poses"
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [1, 2]
        all_features = {}

        for instance_id in tqdm(
            self.instances, disable=(not self.config.show_progress)
        ):
            instance_feat = {}
            for landmark_set in self.config.landmarks:
                filepath = (
                    f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
                )
                instance_feat[landmark_set] = np.load(filepath)[
                    : self.config.sequence_max_length, :, coordinate_indices
                ]
            all_features[instance_id] = instance_feat
        gc.collect()
        return all_features
