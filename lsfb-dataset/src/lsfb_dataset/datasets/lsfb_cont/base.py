import abc
import numpy as np
import pandas as pd
import pickle
from os import path
from .config import LSFBContConfig
from ...utils.datasets import mini_sample, split_cont
from ...utils.target import (
    combine_binary_vectors,
    combine_binary_vectors_with_coarticulation,
)


class LSFBContBase:

    def __init__(self, config=None, **kwargs):
        self.config = LSFBContConfig(**kwargs) if config is None else config
        if self.config.verbose:
            print('-' * 10, 'LSFB CONT DATASET')

        self.videos = self._load_video_list()

        self.targets: list[np.ndarray] = []
        self.labels: list[int] = []
        self.label_frequencies: list[int] = []
        self._load_targets()

    def _load_video_list(self) -> pd.DataFrame:
        split = self.config.split
        videos: pd.DataFrame = pd.read_csv(self.config.video_list_file)
        train_videos, test_videos = split_cont(videos, signers_frac=0.8, seed=self.config.seed)

        if split == 'mini_sample':
            videos = mini_sample(videos, num_samples=10, seed=42)
        elif split == 'train':
            videos = train_videos
        elif split == 'test':
            videos = test_videos
        elif split != 'all':
            raise ValueError(f'Unknown split: {split}.')
        return videos

    def _load_targets(self):
        target = self.config.target
        targets_dir = self.config.targets_dir
        hands = self.config.hands

        if target == 'signs':
            filename = 'binary.pck'
            self.labels = ['waiting', 'signing']
            self.label_frequencies = [0.632736, 0.367264]
        elif target == 'signs_and_transitions':
            filename = 'binary_with_coarticulation.pck'
            self.labels = ['waiting', 'signing', 'coarticulation']
            self.label_frequencies = [0.519183, 0.367264, 0.113553]
        elif target == 'activity':
            filename = 'activity.pck'
            self.labels = ['waiting', 'signing']
            self.label_frequencies = [0.519183, 0.480817]
        else:
            raise ValueError(f'Unknown target: {target}.')

        target_filepath = path.join(targets_dir, filename)
        with open(target_filepath, 'rb') as file:
            target_vectors: dict = pickle.load(file)

        for _, video in self.videos.iterrows():
            filename = video['filename']
            vec = target_vectors.get(filename)

            if hands == 'left':
                vec = vec[0]
            elif hands == 'right':
                vec = vec[1]
            elif hands == 'both':
                if target == 'signs_and_transitions':
                    vec = combine_binary_vectors_with_coarticulation(vec[0], vec[1])
                else:
                    vec = combine_binary_vectors(vec[0], vec[1])

            assert vec is not None, f'Target not found for video {filename}.'
            self.targets.append(vec)

        if self.config.verbose:
            print('Target vectors loaded.')

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass
