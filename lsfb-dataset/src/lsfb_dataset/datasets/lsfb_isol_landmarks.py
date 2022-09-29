import os
import pandas as pd
from tqdm import tqdm

from .types import *
from ..utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from ..utils.datasets import split_isol, mini_sample, create_mask


def _select_videos(videos, lemmes, split: str):
    videos = videos[videos['class'].isin(lemmes.index)]

    if split == 'mini_sample':
        videos = mini_sample(videos)
    elif split != 'all':
        train_videos, test_videos = split_isol(videos)
        if split == 'train':
            videos = train_videos
        elif split == 'test':
            videos = test_videos

    return videos


def _load_landmarks(
        root: str,
        videos: pd.DataFrame,
        landmarks: str,
        hands: str,
        only_selected_hand: bool,
        sequence_max_length: int,
):
    load_pose = True
    load_hands = True
    if landmarks == 'pose':
        load_hands = False
    elif landmarks == 'hands':
        load_pose = False

    hands = hands if only_selected_hand else 'both'
    features = []
    targets = []

    print('Loading features and labels for each isolated sign...')
    progress_bar = tqdm(videos.iterrows(), total=videos.shape[0])
    for index, video in progress_bar:
        data = []
        if load_pose:
            data.append(load_pose_landmarks(root, video['pose']))
        if load_hands:
            data.append(load_hands_landmarks(root, video['hands'], hands))
        data = pd.concat(data, axis=1).values[:sequence_max_length]

        features.append(data)
        targets.append(video['class'])

    return features, targets


class LSFBIsolLandmarks:

    def __init__(
            self,
            root: str,
            *,
            transform=None,
            target_transform=None,
            mask_transform=None,
            lemmes_nb: int = 10,
            lemme_list_path: str = 'lemmes.csv',
            videos_list_path: str = 'videos.csv',
            split: DataSubset = 'all',
            landmarks: LandmarkSet = 'pose',
            hands: Hand = 'both',
            only_selected_hand: bool = False,
            sequence_max_length: int = 50,
            padding: bool = True,
            return_mask: bool = True,
            mask_value: int = 0,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform

        self.sequence_max_length = sequence_max_length
        self.padding = padding
        self.return_mask = return_mask
        self.mask_value = mask_value

        lemme_list_path = os.path.join(root, lemme_list_path)
        videos_list_path = os.path.join(root, videos_list_path)

        lemmes = pd.read_csv(lemme_list_path)
        lemmes = lemmes.iloc[:lemmes_nb]

        self.videos = pd.read_csv(videos_list_path)
        self.videos = _select_videos(self.videos, lemmes, split)

        self.features, self.targets = _load_landmarks(
            root,
            self.videos,
            landmarks,
            hands,
            only_selected_hand,
            sequence_max_length,
        )
        self.labels = lemmes['lemme']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]
        pad_value = 0

        if self.padding:
            pad_value = self.sequence_max_length - len(features)
            features = pad_landmarks(features, pad_value)

        if self.transform is not None:
            features = self.transform(features)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_mask:
            mask = create_mask(self.sequence_max_length, pad_value, self.mask_value)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            return features, target, mask

        return features, target
