import gc
from sys import prefix

from tqdm import tqdm
import numpy as np

from lsfb_dataset.datasets.lsfb_cont.base import LSFBContBase
from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig
from lsfb_dataset.utils.body_parts import get_body_part


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
                window=(1500, 1200)
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
        if self.config.segment_unit == 'ms':
            start, end = start*20, end*20
        annotations = self.annotations[instance_id]
        annotations = annotations.loc[(annotations['end'] >= start) & (annotations['start'] <= end)]
        annotations.loc[:, 'start'] = annotations['start'] - start
        annotations.loc[:, 'end'] = annotations['end'] - start
        features, annotations = self._apply_transforms(features, annotations)
        return features, annotations

    def _load_features(self):
        pose_folder = 'poses_raw' if self.config.use_raw else 'poses'
        coordinate_indices = [0, 1, 2] if self.config.use_3d else [0, 1]
        progress_bar = tqdm(
            self.instances,
            disable=(not self.config.show_progress),
            leave=False,
            unit='instance',
        )
        progress_bar.set_description('Loading features')
        for instance_id in progress_bar:
            instance_features = {}
            for landmarks_set, body_part in self.landmarks_sets:
                filepath = f"{self.config.root}/{pose_folder}/{landmarks_set}/{instance_id}.npy"
                lm_set_features = np.load(filepath)[:, :, coordinate_indices]
                if body_part is not None:
                    lm_set_features = get_body_part(lm_set_features, body_part)
                    instance_features[body_part] = lm_set_features
                else:
                    instance_features[landmarks_set] = lm_set_features
            self.features[instance_id] = instance_features
        gc.collect()
