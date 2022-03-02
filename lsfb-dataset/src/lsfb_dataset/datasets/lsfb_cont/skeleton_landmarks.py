"""
@author ppoitier
@version 1.0
"""

import pandas as pd
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset
import os
from ...utils.annotations import annotations_to_vec, create_coerc_vec
from ...utils.datasets import make_windows, load_data
from tqdm.auto import tqdm
from typing import Optional, Callable


class SkeletonLandmarksDataset(Dataset):
    """
    This class represents a dataset containing landmarks of the upper body skeleton.

    Each item of the dataset is the upper skeleton of an entire video. Therefore, they have different lengths.
    So, the batch_size parameter of the pytorch dataloader does not work with this dataset.

    Skeletons and annotations are loaded on the fly.
    """

    def __init__(self, root_dir: str, video_information: pd.DataFrame, transform=None, isolate_transitions=False):
        """
        Parameters
        ----------
        root_dir : str
            The root directory of the LSFB CONT dataset.
        video_information : pd.Dataframe
            The dataframe containing the information about the videos. See VideoLoader class.
        transform : Callable
            Any transformation applied to the features.
        isolate_transitions : bool
            If true, use 3 classes (waiting, talking, transition).
            Otherwise, use only 2 classes (waiting, talking).
            A transition is less than 1 second between two annotations.
        """
        self.root = root_dir
        self.video_information = video_information[[
            'frames_nb',
            'right_hand_annotations',
            'left_hand_annotations',
            'upper_skeleton'
        ]].dropna()
        self.transform = transform
        self.transitions = isolate_transitions

        if isolate_transitions:
            self.class_names = ['waiting', 'talking', 'transition']
            self.class_proportion = torch.tensor([0.5343, 0.3381, 0.1276])
        else:
            self.class_names = ['waiting', 'talking']
            self.class_proportion = torch.tensor([0.5343, 0.4657])
        self.class_weights = 1 / self.class_proportion

    def __len__(self):
        """
        Return
        ------
        int
            the length of the dataset.
        """
        return self.video_information.shape[0]

    def __getitem__(self, index):
        video = self.video_information.iloc[index]
        features = pd.read_csv(os.path.join(self.root, video['upper_skeleton'])).values

        annot_right = pd.read_csv(os.path.join(self.root, video['right_hand_annotations']))
        annot_left = pd.read_csv(os.path.join(self.root, video['left_hand_annotations']))
        classes = annotations_to_vec(annot_right, annot_left, int(video['frames_nb']))

        if self.transitions:
            classes = create_coerc_vec(classes)

        if self.transform:
            features = self.transform(features)

        return features, classes


# def _make_windows(videos: pd.DataFrame, window_size: int, stride: int):
#     frames = []
#     # (video_idx, start, off)
#
#     for idx, video in videos.iterrows():
#         frames_nb = int(video['frames_nb'])
#         for f in range(0, frames_nb, stride):
#             frames.append((idx, f, f + window_size))
#
#     return frames
#
#
# def _load_data(root: str, videos: pd.DataFrame, isolate_transition=False):
#     print('Loading skeletons and classes...')
#     data = {}
#
#     for idx, video in tqdm(videos.iterrows(), total=videos.shape[0]):
#         skeleton = pd.read_csv(os.path.join(root, video['upper_skeleton'])).values
#
#         annot_right = pd.read_csv(os.path.join(root, video['right_hand_annotations']))
#         annot_left = pd.read_csv(os.path.join(root, video['left_hand_annotations']))
#         classes = annotations_to_vec(annot_right, annot_left, int(video['frames_nb']))
#         if isolate_transition:
#             classes = create_coerc_vec(classes)
#
#         data[idx] = skeleton, classes
#     return data


class SkeletonLandmarksWindowedDataset(Dataset):
    """
    This class represents a dataset containing landmarks of the upper body skeleton.

    A mobile window is used on all the videos. Therefore, each item has a fixed length of {window_size}
    and all the items have the same size.

    Skeletons and annotations are stored in RAM when the dataset is instantiated.
    It requires about 4GB of RAM.
    """

    def __init__(self,
                 root_dir: str, video_information: pd.DataFrame,
                 transform: Optional[Callable] = None,
                 window_size=1500, window_stride=500, isolate_transitions=False):
        """
        Parameters
        ----------
        root_dir : str
            The root directory of the LSFB CONT dataset.
        video_information : pd.Dataframe
            The dataframe containing the information about the videos. See VideoLoader class.
        transform : Callable, optional
            Any transformation applied to the features.
        window_size : int
            The size of the items in the dataset.
        window_stride : int
            The stride length between two windows
        isolate_transitions : bool
            If true, use 3 classes (waiting, talking, transition).
            Otherwise, use only 2 classes (waiting, talking).
            A transition is less than 1 second between two annotations.
        """

        self.root = root_dir
        self.transform = transform

        video_information = video_information[[
            'frames_nb',
            'right_hand_annotations',
            'left_hand_annotations',
            'upper_skeleton'
        ]].dropna()

        self.data = load_data(root_dir, 'upper_skeleton', video_information, isolate_transitions)
        self.window_size = window_size
        self.windows = make_windows(video_information, window_size, window_stride)

        if isolate_transitions:
            self.class_names = ['waiting', 'talking', 'transition']
            self.class_proportion = torch.tensor([0.5343, 0.3381, 0.1276])
        else:
            self.class_names = ['waiting', 'talking']
            self.class_proportion = torch.tensor([0.5343, 0.4657])
        self.class_weights = 1 / self.class_proportion

    def __len__(self):
        """
        Return
        ------
        int
            the length of the dataset.
        """
        return len(self.windows)

    def __getitem__(self, index):
        video_idx, start, end = self.windows[index]
        features, classes = self.data[video_idx]
        features = features[start:end]
        features = torch.from_numpy(features)
        features = pad(features, (0, 0, 0, self.window_size - features.shape[0]))

        if self.transform is not None:
            features = self.transform(features)

        classes = classes[start:end]
        classes = torch.from_numpy(classes).long()
        classes = pad(classes, (0, self.window_size - classes.shape[0]))

        return features, classes


