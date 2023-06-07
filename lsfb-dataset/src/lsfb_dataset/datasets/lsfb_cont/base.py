import abc
import json
import re
from os import path

import numpy as np
import pandas as pd

from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig
from lsfb_dataset.utils.datasets import mini_sample, split_cont, new_split_cont


class LSFBContBase:

    def __init__(self, config=None, **kwargs):
        self.config = LSFBContConfig(**kwargs) if config is None else config
        if self.config.verbose:
            print('-' * 10, 'LSFB CONT DATASET')

        self.videos_metadata = self._load_video_list()
        self.videos: list[str] = self.videos_metadata['filename'].str.replace('.mp4', '').tolist()
        self.targets: dict[str] = {}
        self.target_format = 'segments' if 'segments' in self.config.target else 'vectors'

        self.labels: list[int] = []
        self.label_frequencies: list[int] = []
        self.cls_to_label = {}
        self.label_to_cls = {}

        self.windows = None
        if self.config.window:
            self._make_windows()

        self._load_label_metadata()
        self._load_targets()

    def _load_video_list(self) -> pd.DataFrame:
        split = self.config.split
        videos: pd.DataFrame = pd.read_csv(self.config.video_list_file)

        if split == 'mini_sample':
            videos = mini_sample(videos, num_samples=10, seed=42)
        elif re.match(r'^split \d/\d (train|test)$', split):
            n_splits = int(split[8])
            split_idx = int(split[6])
            is_train = split.endswith('train')
            print(f'Split {split_idx}/{n_splits} - {"train" if is_train else "test"}')
            videos = new_split_cont(videos, n_splits)[split_idx-1][0 if is_train else 1]
        else:
            train_videos, test_videos = split_cont(videos, signers_frac=0.8, seed=self.config.seed)
            if split == 'train':
                videos = train_videos
            elif split == 'test':
                videos = test_videos

        return videos

    def _load_targets(self):
        assert self.config.target in ["gloss_segments", "frame-wise_segmentation"]
        # TODO: subtitle segments
        assert self.config.hands in ['left', 'right', 'both']

        if 'segments' in self.config.target:
            self._load_segments('gloss')
        else:
            raise NotImplementedError('Not implemented yet.')  # TODO

    def _load_segments(self, segment_level: str):
        n_different_signs = self.config.labels
        assert isinstance(self.config.labels, int)
        assert n_different_signs > 0

        MS_PER_FRAME = 1000 // 50

        if self.config.hands == 'both':
            suffix = 'merged'
        else:
            suffix = self.config.hands
        with open(path.join(self.config.root, 'targets', f'{segment_level}_{suffix}.json')) as file:
            segments = json.load(file)

        for video_id in segments:
            video_segments = []
            for segment in segments[video_id]:
                if segment['gloss'] not in self.label_to_cls:
                    continue  # TODO: HANDLE EMPTY GLOSSES IN PARSING
                video_segments.append((
                    segment['start'] if self.config.duration_unit == 'ms' else segment['start'] // MS_PER_FRAME,
                    segment['end'] if self.config.duration_unit == 'ms' else segment['end'] // MS_PER_FRAME,
                    self.label_to_cls[segment['gloss']],
                ))
            self.targets[video_id] = np.array(video_segments, dtype='int32').reshape(-1, 3)

    def _load_label_metadata(self):
        root = self.config.root
        with open(f'{root}/metadata/label_mapping.json') as file:
            self.cls_to_label = json.load(file)
        self.label_to_cls = {v: int(k) for k, v in self.cls_to_label.items()}
        with open(f'{root}/metadata/sign_occurrences.json') as file:
            sign_occurrences = json.load(file)  # TODO

    def _make_windows(self):
        window_size, window_stride = self.config.window
        self.windows = []
        for video_idx, n_frames in enumerate(self.videos_metadata['frames'].tolist()):
            for start in range(0, n_frames, window_stride):
                end = min(start + window_size, n_frames - 1)
                self.windows.append((video_idx, start, end))

    def __getitem__(self, index):
        if self.windows is None:
            return self.__get_instance__(index)
        return self.__get_window__(index)

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __get_instance__(self, index):
        pass

    @abc.abstractmethod
    def __get_window__(self, index):
        pass
