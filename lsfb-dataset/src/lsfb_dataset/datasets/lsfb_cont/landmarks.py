from tqdm import tqdm
import numpy as np
from .base import LSFBContBase
from ...utils.datasets import create_mask
from ...utils.landmarks import (
    load_landmarks,
    pad_landmarks,
)
from ...utils.target import pad_target


class LSFBContLandmarks(LSFBContBase):
    """
    Utility class to load the LSFB CONT Landmarks dataset.
    The dataset must be already downloaded!

    All the landmarks and targets are loaded into memory (RAM) to maximise the efficiency of the training.
    If you don't have enough RAM (between 8GB and 16GB) and don't want to load all of them in memory,
    use the LSFBContLandmarksGenerator class instead.

    Properties:
        features: The list of landmarks for each instance of the dataset
            For each instance i, (L,D) were
                L is the length of the sequence
                D is the number of features depending on the landmarks selected in the configuration
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
            If config is not specified, every needed configuration argument must be manually provided.

    Author: ppoitier
    """

    def __init__(self, **kwargs):
        super(LSFBContLandmarks, self).__init__(**kwargs)

        self.features: list[np.ndarray] = []
        self._load_features()

        # (video_index, window_start, window_end, padding)
        self.windows: list[tuple[int, int, int, int]] = []
        if self.config.window is not None:
            self._build_windows()

    def __len__(self):
        if self.config.window is not None:
            return len(self.windows)
        return len(self.features)

    def __getitem__(self, index):
        features_transform = self.config.features_transform
        target_transform = self.config.target_transform
        transform = self.config.transform

        window = self.config.window
        return_mask = self.config.return_mask
        mask_transform = self.config.mask_transform

        if window is not None:
            landmarks, target, padding = self._get_windowed_item(index)
        else:
            landmarks = self.features[index]
            target = self.targets[index]
            padding = 0

        if features_transform is not None:
            landmarks = features_transform(landmarks)

        if target_transform is not None:
            target = target_transform(target)

        if transform is not None:
            landmarks, target = transform(landmarks, transform)

        if return_mask:
            mask = create_mask(len(target), padding)
            if mask_transform is not None:
                mask = mask_transform(mask)
            return landmarks, target, mask

        return landmarks, target

    def _load_features(self):
        root = self.config.root
        landmarks = self.config.landmarks
        show_progress = self.config.show_progress

        landmarks_nb = self.videos.shape[0]
        landmarks_list = ', '.join(landmarks)

        if self.config.verbose:
            print(f'Loading landmarks {landmarks_list} for {landmarks_nb} videos...')

        progress_bar = tqdm(
            self.videos.iterrows(),
            total=landmarks_nb,
            disable=(not show_progress),
        )

        for _, video in progress_bar:
            self.features.append(load_landmarks(video, root, landmarks))

        if self.config.verbose:
            print('Landmarks loaded.')

    def _build_windows(self):
        window = self.config.window
        window_size, window_stride = window
        for index, landmarks in enumerate(self.features):
            landmarks_nb = landmarks.shape[0]
            for start in range(0, landmarks_nb, window_stride):
                end = min(landmarks_nb, start + window_size)
                padding = window_size - (end - start)
                self.windows.append((index, start, end, padding))

        if self.config.verbose:
            print('Windows successfully created.')

    def _get_windowed_item(self, index: int):
        video_index, start, end, padding = self.windows[index]
        landmarks = pad_landmarks(self.features[video_index][start:end], padding)
        target = pad_target(self.targets[video_index][start:end], padding)
        return landmarks, target, padding
