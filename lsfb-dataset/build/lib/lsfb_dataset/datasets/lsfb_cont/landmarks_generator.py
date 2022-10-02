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

    Configuration args (see LSFBContConfig):
        root: Root directory of the LSFB_CONT dataset.
            The dataset must already be downloaded !
        landmarks: Select which landmarks (features) to use.
            'pose' for pose skeleton (23 landmarks);
            'hands_left' for left hand skeleton (21 landmarks);
            'hands_right' for right hand skeleton (21 landmarks);
            Default = ['pose', 'hand_left', 'hand_right'].

        features_transform: Callable object used to transform the features.
        target_transform: Callable object used to transform the targets.
        transform: Callable object used to transform both the features and the targets.

        split: Specify which subset of the dataset is used.
            'train' for training set;
            'test' for the test set;
            'all' for all the instances of the dataset;
            'mini_sample' for a tiny set of instances.
            Default = 'all'.
        hands: Specify which hands are targeted in the segmentation.
            'right' for the signs from the right hand of the signer;
            'left' for the signs from the left hand of the signer;
            'both' for the signs from both hands of the signer.
            Default = 'both'.
        target: Specify the kind of segmentation used as a target.
            'signs' for the labels waiting and signing only.
            'signs_and_transitions' for the labels waiting, signing and coarticulation.
            'activity' for the labels waiting and signing,
                where intermediate movements are labelled as signing and not waiting anymore.
            Default = 'signs'.

        window: Optional argument (window_size, window_stride) to use fixed-size windows instead of variable length sequences.
            If specified, the dataset is windowed with a window size and a window stride.
            Be careful, this argument changes the number of instances in the dataset !
            Default = None.
        return_mask: If true, return a mask in addition to the features and the target for each instance.
            The mask is filled with 1's and 0's where the padding has been applied.
            Default = False.
        mask_transform: Callable object used to transform the masks.
            You need to set return_mask to `True` to use this transform !

        video_list_file: The filepath of the video list CSV file.
            This filepath is relative to the root of the dataset.
        targets_dir: The path of the directory that contains the target vectors.
            This filepath is relative to the root of the dataset.

        show_progress: If true, shows a progress bar while the dataset is loading.

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

        features = load_landmarks(
            self.videos.iloc[index],
            self.config.root,
            self.config.landmarks
        )
        target = self.targets[index]
        padding = 0

        if window is not None:
            features, target, padding = self._get_windowed_item(index, features, target)

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

    def _get_windowed_item(self, index, features, target):
        video_index, start, end, padding = self.windows[index]
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
