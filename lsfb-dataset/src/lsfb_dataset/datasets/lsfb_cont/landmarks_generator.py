from .base import LSFBContBase
from ...utils.datasets import create_mask
from ...utils.landmarks import load_landmarks, pad_landmarks
from ...utils.target import pad_target


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
            If config is not specified, every needed configuration argument must be manually provided.

    Author: ppoitier
    """

    def __init__(self, **kwargs):
        super(LSFBContLandmarksGenerator, self).__init__(**kwargs)

        # (video_index, window_start, window_end, padding)
        self.windows: list[tuple[int, int, int, int]] = []
        if self.config.window is not None:
            self._build_windows()

    def __len__(self):
        if self.config.window is None:
            return self.videos.shape[0]
        return len(self.windows)

    def __getitem__(self, index):
        features_transform = self.config.features_transform
        target_transform = self.config.target_transform
        transform = self.config.transform

        window = self.config.window
        return_mask = self.config.return_mask
        mask_transform = self.config.mask_transform

        padding = 0

        if window is not None:
            features, target, padding = self._get_windowed_item(index)
        else:
            features = load_landmarks(
                self.videos.iloc[index],
                self.config.root,
                self.config.landmarks
            )
            target = self.targets[index]

        if features_transform is not None:
            features = features_transform(features)

        if target_transform is not None:
            target = target_transform(target)

        if transform is not None:
            features, target = transform(features, transform)

        if return_mask:
            mask = create_mask(len(target), padding)
            if mask_transform is not None:
                mask = mask_transform(mask)
            return features, target, mask

        return features, target

    def _get_windowed_item(self, index):
        video_index, start, end, padding = self.windows[index]
        features = load_landmarks(
            self.videos.iloc[video_index],
            self.config.root,
            self.config.landmarks
        )
        target = self.targets[video_index]

        landmarks = pad_landmarks(features[start:end], padding)
        target = pad_target(target[start:end], padding)
        return landmarks, target, padding

    def _build_windows(self):
        window = self.config.window
        window_size, window_stride = window
        for index, seq_len in enumerate(self.videos['frames']):
            for start in range(0, seq_len, window_stride):
                end = min(seq_len, start + window_size)
                padding = window_size - (end - start)
                self.windows.append((index, start, end, padding))
        print('Windows successfully created.')
