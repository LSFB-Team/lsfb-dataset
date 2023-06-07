import gc
import os

from tqdm import tqdm
import numpy as np

from lsfb_dataset.datasets.lsfb_cont.base import LSFBContBase


class LSFBContLandmarks(LSFBContBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: dict[str, dict[str, np.ndarray]] = {}
        self._load_features()

    def __len__(self):
        if self.config.window is None:
            return len(self.features)
        return len(self.windows)

    def __get_instance__(self, index):
        video_id = self.videos[index]
        target = self.targets[video_id]
        return self.features[video_id], target

    def __get_window__(self, index):
        video_idx, start, end = self.windows[index]
        video_id = self.videos[video_idx]

        features = {lm: lm_feat[start:end] for lm, lm_feat in self.features[video_id].items()}
        target = self.targets[video_id]

        if self.target_format == 'segments':
            if self.config.duration_unit == 'ms':
                start, end = start * 20, end * 20
            target = target[(target[:, 0] < end) & (target[:, 1] > start)]
            target[:, 0] = np.maximum(target[:, 0], start)
            target[:, 1] = np.minimum(target[:, 1], end)
        else:
            raise NotImplementedError('yet to do...')  # TODO

        return features, target

    def _load_features(self):
        for video_id in tqdm(self.videos, disable=(not self.config.verbose)):
            video_feat = {}
            for landmark_set in self.config.landmarks:
                pose_filepath = os.path.join(self.config.root, 'pose_raw', landmark_set, f'{video_id}.npy')
                video_feat[landmark_set] = np.load(pose_filepath)
            self.features[video_id] = video_feat
        gc.collect()
