from typing import Optional, Tuple, List
from os import path
from datetime import datetime

import pandas as pd
import pickle
from tqdm import tqdm
from ..utils.datasets import split_cont, mini_sample, create_mask
from ..utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from ..utils.target import pad_target, combine_binary_vectors, combine_binary_vectors_with_coarticulation
from .types import *


class LSFBContLandmarks:
    """
    LSFB_CONT landmarks Dataset.

    For each video of LSFB_CONT, this dataset provides per-frame 2D-landmarks (skeletons)
    associated with per-frame activity of the signer in the video.

    For each instance (video):
    Features are of size (F, L, 2) where
        - F is the number of frames (or the size of the selected window)
        - L the number of landmarks
    Target is the list of label-index of size F where labels are:
        - waiting (0): the signer is inactive/listening
        - signing (1): the signer is doing a sign
        - coarticulation (2): the signer does an intermediate movement between two signs

    Args:
        root: Root directory of the LSFB_CONT dataset.
            The dataset must already be downloaded.
        transform: Callable object used to transform the features.
        target_transform: Callable object used to transform the targets.
        mask_transform: Callable object used to transform the masks.
            You need to set return_mask to true to use this transform.
        split: Select a specific subset of the dataset. Default = 'all'.
            'train' for training set;
            'test' for the test set;
            'all' for all the instances of the dataset;
            'mini_sample' for a tiny set of instances.
        landmarks: Select which landmarks (features) to use. Default = ['pose', 'hand_left', 'hand_right'].
            'pose' for pose skeleton (23 landmarks);
            'hands_left' for left hand skeleton (21 landmarks);
            'hands_right' for right hand skeleton (21 landmarks);

        hands: Select targeted hands for the signer activity. Default = 'right'.
            'right' for the signs from the right hand of the signer;
            'left' for the signs from the left hand of the signer;
            'both' for the signs from both hands of the signer.
        target: Select which target to use. Default = 'signs'.
            'signs' to use labels waiting and signing only.
            'signs_and_transitions' to use labels waiting, signing and coarticulation.
            'activity' to use labels waiting and signing only with intermediate movements include in signing label.
        window: If not None, the dataset is windowed with (window_size, window_stride).
            This option changes the number of instances in the dataset. Default = None.
        return_mask: If true, return a mask in addition to the features and the target.
            The mask is filled with 1's and 0's where the padding has been applied.
        video_list_file: The path of the video list file.
            If the root is specified, the path is relative.
        targets_dir: The path of the directory containing the target annotations.
            If the root is specified, the path is relative.
        show_progress: If true, show a progress bar while the dataset is loading.

    """
    def __init__(
            self,
            root: str,
            landmarks: Optional[List[str]] = None,
            transform=None,
            target_transform=None,
            mask_transform=None,
            split: DataSubset = 'all',
            hands: Hand = 'both',
            target: Target = 'signs',
            window: Optional[Tuple[int, int]] = None,
            return_mask: bool = False,
            video_list_file: str = 'valid_videos.csv',
            targets_dir: str = 'annotations/vectors',
            show_progress: bool = True,
    ):
        super(LSFBContLandmarks, self).__init__()

        print('-' * 10, 'LSFB CONT DATASET')
        start_time = datetime.now()

        self.landmarks = landmarks
        if self.landmarks is None:
            self.landmarks = ['pose', 'hand_left', 'hand_right']

        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform

        self.root = root
        self.video_list_file = video_list_file
        self.targets_dir = targets_dir
        self.hands = hands
        self.show_progress = show_progress

        if root is not None:
            self.video_list_file = path.join(root, video_list_file)
            self.targets_dir = path.join(root, targets_dir)

        if mask_transform is not None and return_mask is False:
            raise ValueError('Mask transform is defined, but the mask is not returned.')

        self.videos = None
        self.features = []
        self.targets = []

        self.__load_video_metadata(split)
        self.__load_landmarks()
        self.__load_targets(target)

        self.windows = None
        self.return_mask = return_mask
        if window is not None:
            window_size, window_stride = window
            self.windows = []
            self.__make_windows(window_size, window_stride)

        print('-' * 10)
        print('loading time:', datetime.now() - start_time)

    def __len__(self) -> int:
        """
        Return
        ------
        int
            the length of the dataset.
        """
        if self.windows is not None:
            return len(self.windows)

        return len(self.features)

    def __get_windowed_item(self, index):
        video_idx, start, end, padding = self.windows[index]
        landmarks = pad_landmarks(self.features[video_idx][start:end], padding)
        target = pad_target(self.targets[video_idx][start:end], padding)
        return landmarks, target, padding

    def __getitem__(self, index: int):
        if self.windows is not None:
            landmarks, target, padding = self.__get_windowed_item(index)
        else:
            landmarks = self.features[index]
            target = self.targets[index]
            padding = 0

        if self.transform is not None:
            landmarks = self.transform(landmarks)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_mask:
            mask = create_mask(len(target), padding)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            return landmarks, target, mask

        return landmarks, target

    def __load_video_metadata(self, split: str):
        self.videos: pd.DataFrame = pd.read_csv(self.video_list_file)
        if split == 'mini_sample':
            self.videos = mini_sample(self.videos)
        elif split != 'all':
            train_videos, test_videos = split_cont(self.videos, signers_frac=0.6, seed=42)
            if split == 'train':
                self.videos = train_videos
            elif split == 'test':
                self.videos = test_videos
            else:
                raise ValueError(f'Unknown split: {split}.')

    def __load_landmarks(self):
        landmarks_nb = self.videos.shape[0]
        landmarks_list = ', '.join(self.landmarks)
        print(f'Loading landmarks {landmarks_list} for {landmarks_nb} videos...')

        progress_bar = tqdm(self.videos.iterrows(), total=landmarks_nb, disable=(not self.show_progress))
        for _, video in progress_bar:
            video_lm = []
            for lm_type in self.landmarks:
                if lm_type == 'pose':
                    video_lm.append(load_pose_landmarks(self.root, video['pose']))
                elif lm_type == 'hand_left':
                    video_lm.append(load_hands_landmarks(self.root, video['hands'], 'left'))
                elif lm_type == 'hand_right':
                    video_lm.append(load_hands_landmarks(self.root, video['hands'], 'right'))
                else:
                    raise ValueError(f'Unknown landmarks: {lm_type}.')
            self.features.append(pd.concat(video_lm, axis=1).values)

    def __load_targets(self, target: str):

        if target == 'signs':
            filename = 'binary.pck'
        elif target == 'signs_and_transitions':
            filename = 'binary_with_coarticulation.pck'
        elif target == 'activity':
            filename = 'activity.pck'
        else:
            raise ValueError(f'Unknown target: {target}.')

        target_filepath = path.join(self.targets_dir, filename)
        with open(target_filepath, 'rb') as file:
            target_vectors: dict = pickle.load(file)

        for _, video in self.videos.iterrows():
            filename = video['filename']
            vec = target_vectors.get(filename)

            if self.hands == 'left':
                vec = vec[0]
            elif self.hands == 'right':
                vec = vec[1]
            elif self.hands == 'both':
                if target == 'signs_and_transitions':
                    vec = combine_binary_vectors_with_coarticulation(vec[0], vec[1])
                else:
                    vec = combine_binary_vectors(vec[0], vec[1])

            assert vec is not None, f'Target not found for video {filename}.'
            self.targets.append(vec)

        print('Target vectors loaded.')

    def __make_windows(self, window_size: int, window_stride: int):
        for idx, landmarks in enumerate(self.features):
            landmarks_nb = landmarks.shape[0]
            for start in range(0, landmarks_nb, window_stride):
                end = min(landmarks_nb, start + window_size)
                padding = window_size - (end - start)
                self.windows.append((idx, start, end, padding))
        print('Windows successfully created.')
