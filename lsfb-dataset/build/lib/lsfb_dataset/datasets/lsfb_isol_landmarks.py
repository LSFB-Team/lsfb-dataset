import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from .types import *
from typing import Optional, List
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
        landmarks: List[str],
        sequence_max_length: int,
        show_progress: bool,
):
    landmark_list = ', '.join(landmarks)
    print(f'Loading features ({landmark_list}) and labels for each isolated sign...')

    features = []
    targets = []

    progress_bar = tqdm(videos.iterrows(), total=videos.shape[0], disable=(not show_progress))
    for index, video in progress_bar:
        data = []

        for lm_type in landmarks:
            if lm_type == 'pose':
                data.append(load_pose_landmarks(root, video['pose']))
            elif lm_type == 'hand_left':
                data.append(load_hands_landmarks(root, video['hands'], 'left'))
            elif lm_type == 'hand_right':
                data.append(load_hands_landmarks(root, video['hands'], 'right'))
            else:
                raise ValueError(f'Unknown landmarks: {lm_type}.')

        features.append(pd.concat(data, axis=1).values[:sequence_max_length])
        targets.append(video['class'])

    return features, targets


class LSFBIsolLandmarks:

    """
    LSFB_ISOL landmarks Dataset.

    For each clips of LSFB_ISOL, this dataset provides a per-frame 2D-landmarks (skeleton)
    associated with a label for the performed sign (gloss).

    For each instance (video):
    **Features** are of size (F, L, 2) where
        - F is the number of frames
        - L the number of landmarks
    **Target** is a string representing the name of the gloss performed.

    Args:
        root: Root directory of the LSFB_ISOL dataset.
            The dataset must already be downloaded.
        landmarks: Select which landmarks (features) to use. Default = ['pose', 'hand_left', 'hand_right'].
            'pose' for pose skeleton (23 landmarks);
            'hands_left' for left hand skeleton (21 landmarks);
            'hands_right' for right hand skeleton (21 landmarks);
        transform: Callable object used to transform the features.
        target_transform: Callable object used to transform the targets.
        mask_transform: Callable object used to transform the masks.
            You need to set return_mask to true to use this transform.
        lemmes_nb: The minimal number of examples per lemmes. Default=10
        lemme_list_path: Path to the csv containing the lemmes lists. Default="lemmes.csv"
        videos_list_path: Path to the csv containing the video information. Default="clips.csv"
        split: Select a specific subset of the dataset. Default = 'all'.
            'train' for training set;
            'test' for the test set;
            'all' for all the instances of the dataset;
            'mini_sample' for a tiny set of instances.
        sequence_max_length: Max lenght of the clip sequence. Default=50.
        padding: Pad all sequence to the same length.
        return_mask: Returning padding mask for the sequence.
        mask_value: Value of the masked part of the clips.
        show_progress: If true, show a progress bar while the dataset is loading.

    """

    def __init__(
            self,
            root: str,
            landmarks: Optional[List[str]] = None,
            transform=None,
            target_transform=None,
            mask_transform=None,
            lemmes_nb: int = 10,
            lemme_list_path: str = 'lemmes.csv',
            videos_list_path: str = 'clips.csv',
            split: DataSubset = 'all',
            sequence_max_length: int = 50,
            padding: bool = True,
            return_mask: bool = True,
            mask_value: int = 0,
            show_progress=True,
    ):
        print('-'*10, 'LSFB ISOL DATASET')
        start_time = datetime.now()

        if landmarks is None:
            landmarks = ['pose', 'hand_left', 'hand_right']

        self.landmarks = landmarks
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

        print(lemmes)

        self.videos = pd.read_csv(videos_list_path)
        self.videos = _select_videos(self.videos, lemmes, split)

        self.features, self.targets = _load_landmarks(
            root,
            self.videos,
            landmarks,
            sequence_max_length,
            show_progress,
        )
        self.labels = lemmes['lemme']

        print('-'*10)
        print('loading time:', datetime.now() - start_time)

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
