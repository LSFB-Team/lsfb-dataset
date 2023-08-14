import numpy as np

from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase


class LSFBIsolLandmarksGenerator(LSFBIsolBase):
    """
        Utility class to load the LSFB ISOL Landmarks dataset.
        The dataset must be already downloaded!

        All the landmarks and targets are lazily loaded.
        Therefore, iterating over all the instances can be a bit slow.
        If you have enough RAM and want faster iterations, use the `LSFBIsolLandmarks` class instead.

        Example:
            ```python
            my_dataset_config = LSFBIsolConfig(
                root="./my_dataset",
                split="fold_1",
                n_labels=750,
                target='sign_gloss',
                sequence_max_length=10,
                use_3d=True,
            )

            my_dataset = LSFBIsolLandmarksGenerator(my_dataset_config)
            features, target = dataset[30]
            ```

        Args:
            config: The configuration object (see `LSFBContConfig`).

        Author:
            ppoitier (v 2.0)
        """

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
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [0, 1]
        max_len = self.config.sequence_max_length
        instance_features = {}
        for landmark_set in self.config.landmarks:
            pose_path = f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
            lm_set_features = np.load(pose_path)[:, :, coordinate_indices]
            if max_len is not None:
                lm_set_features = lm_set_features[:max_len]
            instance_features[instance_id] = lm_set_features
        return instance_features
