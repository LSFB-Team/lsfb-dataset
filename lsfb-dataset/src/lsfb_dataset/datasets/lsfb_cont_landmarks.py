from typing import Optional, Tuple
from os import path

import pandas as pd
import pickle
from tqdm import tqdm
from ..utils.datasets import split_cont, mini_sample, create_mask
from ..utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from ..utils.target import pad_target
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
        TODO : UPDATE docstring
        root: Root directory of the LSFB_CONT dataset.
            The dataset must already be downloaded.
        split: Select a specific subset of the dataset. Default = 'all'.
            'train' for training set;
            'test' for the test set;
            'all' for all the instances of the dataset.
        landmarks: Select which landmarks (features) to use. Default = 'pose'.
            'pose' for pose skeleton (23 landmarks);
            'hands' for hand skeleton (21 landmarks per hand);
            'all' the all the landmarks.
        hands: Select targeted hands for the signer activity. Default = 'right'.
            'right' for the signs from the right hand of the signer;
            'left' for the signs from the left hand of the signer;
            'both' for the signs from both hands of the signer.
        only_selected_hands: If true, the landmarks of the non-targeted hand are not used.
            Default = False.
        target: Select which target to use. Default = 'signs'.
            'signs' to use labels waiting and signing only.
            'signs_and_transitions' to use labels waiting, signing and coarticulation.
            'activity' to use labels waiting and signing only with intermediate movements include in signing label.
        window: If not None, the dataset is windowed with (window_size, window_stride).
            This option changes the number of instances in the dataset. Default = None.
        video_list_file: The path of the video list file.
            If the root is specified, the path is relative.
        targets_dir: The path of the directory containing the target annotations.
            If the root is specified, the path is relative.

    """
    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            mask_transform=None,
            split: DataSubset = 'all',
            landmarks: LandmarkSet = 'all',
            hands: Hand = 'both',
            target: Target = 'signs',
            only_selected_hands: bool = True,
            window: Optional[Tuple[int, int]] = None,
            return_mask: bool = False,
            video_list_file: str = 'valid_videos.csv',
            targets_dir: str = 'annotations/hands',
    ):
        super(LSFBContLandmarks, self).__init__()
        assert split in ['all', 'train', 'test', 'mini_sample'], f'Unknown subset of the dataset: {split}'
        assert landmarks in ['pose', 'hands', 'all'], f'Unknown landmarks: {landmarks}'
        assert hands in ['right', 'left', 'both'], f'Unknown hands: {hands}'
        assert target in ['activity', 'signs', 'signs_and_transitions'], f'Unknown target: {target}'
        assert not (mask_transform is not None and return_mask is False),\
            f'Mask transform is defined, but the mask is not returned.'

        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform

        self.root = root
        self.video_list_file = video_list_file
        self.targets_dir = targets_dir
        self.hands = hands
        self.only_selected_hands = only_selected_hands

        if root is not None:
            self.video_list_file = path.join(root, video_list_file)
            self.targets_dir = path.join(root, targets_dir)

        self.videos = None
        self.targets = []
        self.landmarks = []
        self.landmark_types = []

        self.__load_video_metadata(split)
        self.__load_targets(target)
        self.__load_landmarks(landmarks)

        self.windows = None
        self.return_mask = return_mask
        if window is not None:
            window_size, window_stride = window
            self.windows = []
            self.__make_windows(window_size, window_stride)

        landmarks_nb = len(self.landmarks)
        targets_nb = len(self.targets)
        if landmarks_nb != targets_nb:
            raise ValueError(f'Different number of landmarks ({landmarks_nb}) and targets ({targets_nb}).')

    def __len__(self) -> int:
        """
        Return
        ------
        int
            the length of the dataset.
        """
        if self.windows is not None:
            return len(self.windows)

        return len(self.landmarks)

    def __get_windowed_item(self, index):
        video_idx, start, end, padding = self.windows[index]
        landmarks = pad_landmarks(self.landmarks[video_idx][start:end], padding)
        target = pad_target(self.targets[video_idx][start:end], padding)
        return landmarks, target, padding

    def __getitem__(self, index: int):
        if self.windows is not None:
            landmarks, target, padding = self.__get_windowed_item(index)
        else:
            landmarks = self.landmarks[index]
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

    def __load_landmarks(self, landmark_kind: str):
        landmarks_nb = self.videos.shape[0]
        print(f'Loading {landmark_kind} landmarks for {landmarks_nb} videos...')

        load_pose = False
        load_hands = False

        if landmark_kind == 'pose':
            load_pose = True
        elif landmark_kind == 'hands':
            load_hands = True
        elif landmark_kind == 'all':
            load_pose = True
            load_hands = True

        hands = self.hands if self.only_selected_hands else 'both'

        if load_pose:
            self.landmark_types.append('pose')
        if load_hands:
            if hands == 'both':
                self.landmark_types.append('hand_left')
                self.landmark_types.append('hand_right')
            else:
                self.landmark_types.append(f'hand_{hands}')

        progress_bar = tqdm(self.videos.iterrows(), total=landmarks_nb)
        for _, video in progress_bar:
            video_lm = []
            if load_pose:
                video_lm.append(load_pose_landmarks(self.root, video['pose']))
            if load_hands:
                video_lm.append(load_hands_landmarks(self.root, video['hands'], hands))
            video_lm = pd.concat(video_lm, axis=1)
            self.landmarks.append(video_lm.values)

        print('Landmarks successfully loaded.')

    def __load_targets(self, target: str):
        if target == 'signs':
            filename = 'binary.pck'
        elif target == 'activity':
            filename = 'binary_activity.pck'
        elif target == 'signs_and_transitions':
            filename = 'binary_with_coarticulation.pck'
        else:
            raise ValueError(f'Unknown target: {target}.')

        target_filepath = path.join(self.targets_dir, self.hands, filename)
        with open(target_filepath, 'rb') as file:
            target_dict: dict = pickle.load(file)

        for _, video in self.videos.iterrows():
            filename = video['filename']
            vec = target_dict.get(filename)
            assert vec is not None, f'Target not found for video {filename}.'
            self.targets.append(vec)

        print('Targets successfully loaded.')

    def __make_windows(self, window_size: int, window_stride: int):
        for idx, landmarks in enumerate(self.landmarks):
            landmarks_nb = landmarks.shape[0]
            for start in range(0, landmarks_nb, window_stride):
                end = min(landmarks_nb, start + window_size)
                padding = window_size - (end - start)
                self.windows.append((idx, start, end, padding))
        print('Windows successfully created.')
