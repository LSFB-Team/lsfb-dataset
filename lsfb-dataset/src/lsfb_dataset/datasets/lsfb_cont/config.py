from dataclasses import dataclass
from os import path
from typing import List, Optional, Tuple, Callable
from ..types import *


@dataclass
class LSFBContConfig:
    """
    Simple configuration class for the LSFB_CONT Dataset.

    LSFB: French Belgian Sign Language
    CONT: Continuous videos in LSFB

    See: LSFBContLandmarks

    Args:
        root: Root directory of the LSFB_CONT dataset.
            The dataset must already be downloaded !
        landmarks: Select which landmarks (features) to use.
            'pose' for pose skeleton (23 landmarks);
            'hand_left' for left hand skeleton (21 landmarks);
            'hand_right' for right hand skeleton (21 landmarks);
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
        seed: Seed used to determine the split. Default = 42.
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

        window: Optional argument (window_size, window_stride) to use fixed-size windows instead of
            variable length sequences.
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

        show_progress: If true, shows a progress bar while the dataset is loading. Default = True.
        verbose: If true, print more information about the loading process. Default = True.

    Author: ppoitier
    """

    root: str
    landmarks: Optional[List[str]] = None

    features_transform: Callable = None
    target_transform: Callable = None
    transform: Callable = None

    split: DataSubset = 'all'
    seed: int = 42
    hands: Hand = 'both'
    target: Target = 'signs'

    window: Optional[Tuple[int, int]] = None
    return_mask: bool = False
    mask_transform: Callable = None

    video_list_file: str = 'videos.csv'
    targets_dir: str = 'annotations/vectors'

    show_progress: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = ['pose', 'hand_left', 'hand_right']

        self.video_list_file = path.join(self.root, self.video_list_file)
        self.targets_dir = path.join(self.root, self.targets_dir)

        self.validate()

    def validate(self):
        if self.mask_transform is not None and self.return_mask is False:
            raise ValueError('Mask transform is defined, but the mask is not returned.')
