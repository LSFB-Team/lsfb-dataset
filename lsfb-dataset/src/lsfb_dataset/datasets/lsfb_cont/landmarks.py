from typing import Optional, Tuple, Literal, NewType
from os import path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from tqdm import tqdm
from ...utils.numpy import fill_gaps
from ...utils.datasets import train_split_videos
import pickle

# Dataset subsets
DataSubset = NewType('DataSubset', Literal['all', 'training', 'validation'])

# Landmarks choice
LandmarkSet = NewType('LandmarkSet', Literal['skeleton', 'hand', 'all'])

# Hands choice
Hand = NewType('Hand', Literal['right', 'left', 'both'])

# Target choice
Target = NewType('Target', Literal['activity', 'signs', 'signs_and_transitions'])


class LandmarksDataset(Dataset):
    def __init__(
            self,
            root='./data',
            subset: DataSubset = 'all',
            landmarks: LandmarkSet = 'skeleton',
            hands: Hand = 'right',
            target: Target = 'signs',
            video_file='videos.csv',
            targets_file='annotations.pkl',
            skeletons_dir='features/upper_skeleton',
            hands_dir='features/cleaned_hands',
            masked=False,
            only_selected_hands=True,
            window: Optional[Tuple[int, int]] = None,
    ):
        super(LandmarksDataset, self).__init__()
        assert subset in ['all', 'training', 'validation'], f'Unknown subset of the dataset: {subset}'
        assert landmarks in ['skeleton', 'hand', 'all'], f'Unknown landmarks: {landmarks}'
        assert hands in ['right', 'left', 'both'], f'Unknown hands: {hands}'
        assert target in ['activity', 'signs', 'signs_and_transitions'], f'Unknown target: {target}'

        self.root = root
        self.video_file = video_file
        self.targets_file = targets_file
        self.skeletons_dir = skeletons_dir
        self.hands_dir = hands_dir
        self.hands = hands
        self.only_selected_hands = only_selected_hands

        if root is not None:
            self.video_file = path.join(root, video_file)
            self.targets_file = path.join(root, targets_file)
            self.skeletons_dir = path.join(root, skeletons_dir)
            self.hands_dir = path.join(root, hands_dir)

        self.videos: pd.DataFrame = pd.read_csv(self.video_file)

        if subset != 'all':
            train_videos, val_videos = train_split_videos(self.videos, signers_frac=0.6, seed=42)
            if subset == 'training':
                self.videos = train_videos
            elif subset == 'validation':
                self.videos = val_videos

        self.masks = []
        self.targets = []
        self.landmarks = []

        self._load_targets(hands, target, masked)
        self._load_landmarks(landmarks, masked)

        self.windows = None
        if window is not None:
            window_size, window_stride = window
            self.windows = []
            self._load_windows(window_size, window_stride)

        assert len(self.landmarks) == len(self.targets), 'Different number of landmarks and targets.'

    def __len__(self):
        """
        Return
        ------
        int
            the length of the dataset.
        """
        if self.windows is not None:
            return len(self.windows)

        return len(self.landmarks)

    def __getitem__(self, index: int):

        if self.windows is None:
            landmarks = self.landmarks[index]
            target = self.targets[index]

            return landmarks, target

        video_idx, start, end, padding = self.windows[index]
        landmarks = self.landmarks[video_idx][start:end]
        target = self.targets[video_idx][start:end].long()
        if padding > 0:
            landmarks = pad(landmarks, (0, 0, 0, padding))
            target = pad(target, (0, padding))
        return landmarks, target

    def _load_landmarks(self, landmark_kind: str, masked: bool):
        load_hands = True
        load_skeleton = True

        if landmark_kind == 'skeleton':
            load_hands = False
        elif landmark_kind == 'hand':
            load_skeleton = False

        landmarks_nb = self.videos.shape[0]
        print(f'Loading {landmarks_nb} {landmark_kind} landmarks...')
        for idx, (_, video) in enumerate(tqdm(self.videos.iterrows(), total=landmarks_nb)):
            filename = video['filename']

            df_hands = None
            df_skeleton = None

            if load_hands:
                df_hands = pd.read_csv(path.join(self.hands_dir, f'{filename}_hands.csv'))
                if self.only_selected_hands:
                    if self.hands == 'right':
                        df_hands = df_hands.loc[:, df_hands.columns.str.startswith('RIGHT')]
                    elif self.hands == 'left':
                        df_hands = df_hands.loc[:, df_hands.columns.str.startswith('LEFT')]
            if load_skeleton:
                df_skeleton = pd.read_csv(path.join(self.skeletons_dir, f'{filename}_skeleton.csv'))

            if df_hands is not None and df_skeleton is not None:
                assert df_hands.shape[0] == df_skeleton.shape[0]
                df = pd.concat([df_hands, df_skeleton], axis=1)
            elif df_hands is not None:
                df = df_hands
            else:
                df = df_skeleton

            values = torch.from_numpy(df.values).float()

            if masked:
                mask = self.masks[idx].unsqueeze(-1).expand_as(values)
                values = torch.masked_select(values, mask).view(-1, values.size(1))

            self.landmarks.append(values)

        print('Landmarks successfully loaded.')

    def _load_targets(self, hands: str, target: str, masked: bool):
        targets_nb = self.videos.shape[0]
        print(f'Loading {targets_nb} targets...')

        with open(self.targets_file, 'rb') as file:
            target_dict: dict = pickle.load(file)

        for _, video in self.videos.iterrows():
            filename = video['filename']
            video_targets = target_dict.get(filename)
            assert video_targets is not None, f'Target not found for video {filename}.'

            mask = torch.from_numpy(video_targets['mask'])

            if target == 'activity':
                self.targets.append(mask)
            else:
                r_hand = video_targets['right_hand']
                l_hand = video_targets['right_hand']

                vec = r_hand

                if hands == 'left':
                    vec = l_hand
                elif hands == 'both':
                    vec = (r_hand | l_hand)

                if target == 'signs_and_transitions':
                    vec = fill_gaps(vec.astype(int), max_gap=50, no_gap=1, fill_with=2)

                vec = torch.from_numpy(vec)
                if masked:
                    vec = torch.masked_select(vec, mask)
                    self.masks.append(mask)

                self.targets.append(vec)

        print('Targets successfully loaded.')

    def _load_windows(self, window_size: int, window_stride: int):
        print('Creating windows...')

        for idx, landmarks in enumerate(self.landmarks):
            landmarks_nb = landmarks.shape[0]
            for start in range(0, landmarks_nb, window_stride):
                end = min(landmarks_nb, start + window_size)
                padding = window_size - (end - start)
                self.windows.append((idx, start, end, padding))
        print('Windows successfully created.')
