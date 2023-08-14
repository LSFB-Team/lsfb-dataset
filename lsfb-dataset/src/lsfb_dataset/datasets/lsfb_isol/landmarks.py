import gc

import numpy as np
from tqdm import tqdm

from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from lsfb_dataset.datasets.lsfb_isol.base import LSFBIsolBase


class LSFBIsolLandmarks(LSFBIsolBase):
    """
    Utility class to load the LSFB ISOL Landmarks dataset.
    The dataset must be already downloaded!

    All the landmarks and targets are loaded in memory.
    Therefore, iterating over all the instances is fast but consumes a lot of RAM.
    If you don't have enough RAM, use the `LSFBIsolLandmarksGenerator` class instead.

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

        my_dataset = LSFBIsolLandmarks(my_dataset_config)
        features, target = dataset[30]
        ```

    If you did not download the dataset, see `lsfb_dataset.Downloader`.

    Args:
        config: The configuration object (see `LSFBContConfig`).

    Author:
        ppoitier (v 2.0)
    """
    # TODO: add class properties to docstring

    def __init__(self, config: LSFBIsolConfig):
        super().__init__(config)
        self.features: dict[str, dict[str, np.ndarray]] = self._load_features()

    def __getitem__(self, index):
        instance_id = self.instances[index]
        features = self.features[instance_id]
        target = self.targets[instance_id]

        if self.config.transform is not None:
            features = self.config.transform(features)

        return features, target

    def _load_features(self):
        pose_folder = "poses_raw" if self.config.use_raw else "poses"
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [0, 1]
        all_features = {}
        max_len = self.config.sequence_max_length

        for instance_id in tqdm(self.instances, disable=(not self.config.show_progress)):
            instance_features = {}
            for landmark_set in self.config.landmarks:
                filepath = f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
                lm_set_features = np.load(filepath)[:, :, coordinate_indices]
                if max_len is not None:
                    lm_set_features = lm_set_features[:max_len]
                instance_features[landmark_set] = lm_set_features
            all_features[instance_id] = instance_features
        gc.collect()
        return all_features
