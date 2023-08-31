import gc
from math import ceil, floor

from tqdm import tqdm
import numpy as np

from lsfb_dataset.datasets.lsfb_cont.base import LSFBContBase
from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig


class LSFBContLandmarks(LSFBContBase):
    """
        Utility class to load the LSFB CONT Landmarks dataset.
        The dataset must be already downloaded!

        All the landmarks and targets are loaded in memory.
        Therefore, iterating over all the instances is fast but consumes a lot of RAM.
        If you don't have enough RAM, use the `LSFBContLandmarksGenerator` class instead.

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

            my_dataset = LSFBContLandmarks(my_dataset_config)
            features, target_annotations = dataset[10]
            ```

        If you did not download the dataset, see `lsfb_dataset.Downloader`.

        Args:
            config: The configuration object (see `LSFBContConfig`).

        Author:
            ppoitier (v 2.0)
    """
    # TODO: add class properties to docstring

    def __init__(self, config: LSFBContConfig):
        super().__init__(config)
        self.features: dict[str, dict[str, np.ndarray]] = {}
        self._load_features()

    def __get_instance__(self, index):
        instance_id = self.instances[index]
        features = self.features[instance_id]
        annotations = self.annotations[instance_id].values
        features, annotations = self._apply_transforms(features, annotations)
        return features, annotations

    def __get_window__(self, index):
        instance_id, start, end = self.windows[index]
        features = {lm: lm_feat[start:end] for lm, lm_feat in self.features[instance_id].items()}

        annotations = self.annotations[instance_id]
        if self.config.segment_unit == 'ms':
            annotations = annotations.loc[
                ((annotations['end'] / 20) >= start) &
                ((annotations['start'] / 20) <= end)
            ]
            annotations.loc[:, 'start'] = annotations['start'] - start * 20
            annotations.loc[:, 'end'] = annotations['end'] - start * 20
        if self.config.segment_unit == 'frame':
            annotations = annotations.loc[
                (annotations['end'] >= start) &
                (annotations['start'] <= end)
            ]
            annotations.loc[:, 'start'] = annotations['start'] - start
            annotations.loc[:, 'end'] = annotations['end'] - start
        else:
            raise ValueError(f'Unknown segment unit: {self.config.segment_unit}.')
        features, annotations = self._apply_transforms(features, annotations)
        return features, annotations

    def _load_features(self):
        pose_folder = 'poses_raw' if self.config.use_raw else 'poses'
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [0, 1]
        for instance_id in tqdm(self.instances, disable=(not self.config.show_progress)):
            instance_feat = {}
            for landmark_set in self.config.landmarks:
                filepath = f"{self.config.root}/{pose_folder}/{landmark_set}/{instance_id}.npy"
                instance_feat[landmark_set] = np.load(filepath)[:, :, coordinate_indices]
            self.features[instance_id] = instance_feat
        gc.collect()
