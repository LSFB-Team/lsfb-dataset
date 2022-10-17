import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from lsfb_dataset.datasets.types import *
from typing import Optional, List
from lsfb_dataset.utils.landmarks import load_pose_landmarks, load_hands_landmarks, pad_landmarks
from lsfb_dataset.utils.datasets import split_isol, mini_sample, create_mask
import abc


def _select_videos(videos, lemmes, split: str):
    videos = videos[videos['lemme'].isin(lemmes.lemme)]

    if split == 'mini_sample':
        videos = mini_sample(videos)
    elif split != 'all':
        train_videos, test_videos = split_isol(videos)
        if split == 'train':
            videos = train_videos
        elif split == 'test':
            videos = test_videos

    return videos


class LSFBIsolBase:

    """
    LSFB_ISOL Base Dataset.

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
        lemmes_nb: Number of lemme to consider. Default=10
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
        if landmarks is None:
            landmarks = ['pose', 'hand_left', 'hand_right']

        self.root = root
        self.landmarks = landmarks
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform

        self.sequence_max_length = sequence_max_length
        self.padding = padding
        self.return_mask = return_mask
        self.mask_value = mask_value
        self.show_progress = show_progress

        lemme_list_path = os.path.join(root, lemme_list_path)
        videos_list_path = os.path.join(root, videos_list_path)

        lemmes = pd.read_csv(lemme_list_path)
        self.lemmes = lemmes.iloc[:lemmes_nb]

        self.videos = pd.read_csv(videos_list_path)
        self.videos = _select_videos(self.videos, lemmes, split)


    @abc.abstractmethod
    def __len__(self):
        pass
        
    @abc.abstractmethod
    def __getitem__(self, index):
        pass
        
