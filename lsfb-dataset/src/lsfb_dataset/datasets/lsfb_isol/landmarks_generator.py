import numpy as np

from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase
import numpy as np


class LSFBIsolLandmarksGenerator(LSFBIsolBase):

    def __init__(self, config: LSFBIsolConfig):
        super().__init__(config)

    def __getitem__(self, index):
        instance_id = self.instances[index]
        features = self._load_instance_features(instance_id)
        target = self.targets[instance_id]

        if self.config.transform is not None:
            features = self.config.transform(features)

        return features, target

    def _load_instance_features(self, instance_id):
        pose_folder = 'poses_raw' if self.config.use_raw else 'poses'
        instance_features = {}
        for landmark_set in self.config.landmarks:
            pose_path = f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
            instance_features[instance_id] = np.load(pose_path)
        return instance_features
