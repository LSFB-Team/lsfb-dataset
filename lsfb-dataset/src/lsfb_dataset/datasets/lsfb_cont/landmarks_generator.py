import numpy as np

from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig
from lsfb_dataset.datasets.lsfb_cont.base import LSFBContBase


class LSFBContLandmarksGenerator(LSFBContBase):
    """
    Utility class to load the LSFB CONT Landmarks dataset.
    The dataset must be already downloaded!

    All the landmarks and targets are lazy loaded. In consequence, iterate over all the instances is slower
    but consumes less memory (RAM).

    If you have enough RAM (more than 16GB) and want more efficient iterations,
    use the LSFBContLandmarks class instead.

    Properties:
        targets: The list of target segmentations for each instance of the dataset

        labels: The targeted labels in the dataset. Example: waiting, signing and coarticulation.
            The labels depend on the targeted segmentation.
        label_frequencies: The frequency of each label used in the dataset.
            The labels depend on the targeted segmentation.

        windows: List of every window in the dataset
            (instance_index, window_start, window_end, padding)
            This list only contains windows if the `window` configuration is set!

    Args:
        config: The configuration object (see LSFBContConfig).

    Author:
        ppoitier (v 2.0)
    """

    def __init__(self, config: LSFBContConfig):
        super(LSFBContLandmarksGenerator, self).__init__(config)

    def __get_instance__(self, index):
        instance_id = self.instances[index]
        features = self._load_instance_features(instance_id)
        annotations = self.annotations[instance_id]
        features, annotations = self._apply_transforms(features, annotations)
        return features, annotations

    def __get_window__(self, index):
        instance_id, start, end = self.windows[index]
        features = self._load_instance_features(instance_id)
        features = {lm: lm_feat[start:end] for lm, lm_feat in features[instance_id].items()}

        # TODO: fix annotation and check with windows
        annotations = self.annotations[instance_id]
        features, annotations = self._apply_transforms(features, annotations)
        return features, annotations

    def _load_instance_features(self, instance_id: str):
        pose_folder = 'poses_raw' if self.config.use_raw else 'poses'
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [1, 2]
        features = {}
        for landmark_set in self.config.landmarks:
            filepath = f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
            features[landmark_set] = np.load(filepath)[:, :, coordinate_indices]
        return features
