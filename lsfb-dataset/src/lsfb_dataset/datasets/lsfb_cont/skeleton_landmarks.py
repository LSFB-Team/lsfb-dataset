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
        self.transform = transform

        video_information = video_information[[
            'frames_nb',
            'right_hand_annotations',
            'left_hand_annotations',
            'upper_skeleton'
        ]].dropna()

        self.data = list(load_data(root_dir, 'upper_skeleton', video_information, isolate_transitions).values())

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
        return len(self.data)

    def __getitem__(self, index):
        features, classes = self.data[index]
        features = torch.from_numpy(features).float()
        classes = torch.from_numpy(classes).long()

        if self.transform is not None:
            features = self.transform(features)

        return features, classes


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
        features = torch.from_numpy(features).float()
        features = pad(features, (0, 0, 0, self.window_size - features.shape[0]))

        if self.transform is not None:
            features = self.transform(features)

        classes = classes[start:end]
        classes = torch.from_numpy(classes).long()
        classes = pad(classes, (0, self.window_size - classes.shape[0]))

        return features, classes


