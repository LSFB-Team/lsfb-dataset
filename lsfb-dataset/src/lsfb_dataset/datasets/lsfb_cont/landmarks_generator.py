import numpy as np

from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig
from lsfb_dataset.datasets.lsfb_cont.base import LSFBContBase


class LSFBContLandmarksGenerator(LSFBContBase):
    """
        Utility class to load the LSFB CONT Landmarks dataset.
        The dataset must be already downloaded!

        All the landmarks and targets are lazily loaded.
        Therefore, iterating over all the instances can be a bit slow.
        If you have enough RAM and want faster iterations, use the `LSFBContLandmarks` class instead.

        Example:
            ```python
            my_dataset_config = LSFBContConfig(
                root="./my_dataset",
                landmarks=['pose', 'left_hand', 'right_hand'],
                split="fold_1",
                n_labels=750,
                segment_level='signs',
                segment_unit='frame',
                segment_label='sign_gloss',
                use_3d=True,
                window=(1500, 1200),
            )

            my_dataset = LSFBContLandmarksGenerator(my_dataset_config)
            features, target_annotations = dataset[10]
            ```

        If you did not download the dataset, see `lsfb_dataset.Downloader`.

        Args:
            config: The configuration object (see `LSFBContConfig`).

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

        annotations = self.annotations[instance_id]
        if self.config.segment_unit == 'ms':
            annotations = annotations.loc[
                ((annotations['end'] / 20) >= start) &
                ((annotations['start'] / 20) <= end)
            ]
        else:
            annotations = annotations.loc[
                (annotations['end'] >= start) &
                (annotations['start'] <= end)
            ]
        annotations.loc[:, 'start'] = annotations['start'] - start
        annotations.loc[:, 'end'] = annotations['end'] - start
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
