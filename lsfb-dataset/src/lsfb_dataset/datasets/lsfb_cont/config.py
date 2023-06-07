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
            'left_hand' for left hand skeleton (21 landmarks);
            'right_hand' for right hand skeleton (21 landmarks);
            Default = ['pose', 'left_hand', 'right_hand'].
        use_3d: If true, use 3D landmarks. Otherwise, use 2D landmarks.
            Default=False.

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
            Default = 'right'.
        target: Specify the format of the target.
            'gloss_segments'
            'subtitle_segments'
            'frame-wise_segmentation'
            Default = 'gloss_segments'
        labels: Specify which label to use for each segment/frame.
            If the argument is an integer `x`, then `x+1` labels are used for the `x` most frequent signs and
            the placeholder label.
            Otherwise, the valid arguments are:
            'binary' for binary frame-wise segmentation with labels `0` for no-sign and `1` for sign;
            'binary_with_coarticulation' for binary frame-wise segmentation where
                short periods between signs have the coarticulation label `2`.
            Default=750
        duration_unit: Specify which unit is used for the boundaries (start, end) of the segments.
            'frames': frame indices in segments boundaries.
            'ms': milliseconds in segments boundaries.
            default='milliseconds'

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

        show_progress: If true, shows a progress bar while the dataset is loading. Default = True.
        verbose: If true, print more information about the loading process. Default = True.

    Author: ppoitier
    """

    root: str
    landmarks: Optional[List[str]] = None
    use_3d: bool = False

    features_transform: Callable = None
    target_transform: Callable = None
    transform: Callable = None

    split: DataSubset = 'all'
    seed: int = 42

    hands: Hand = 'right'
    target: Target = 'gloss_segments'
    labels: int | str = 750
    duration_unit: str = 'ms'

    window: Optional[Tuple[int, int]] = None
    return_mask: bool = False
    mask_transform: Callable = None

    show_progress: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = ['pose', 'left_hand', 'right_hand']

        self.video_list_file = path.join(self.root, 'videos.csv')
        self.validate()

    def validate(self):
        if self.mask_transform is not None and self.return_mask is False:
            raise ValueError('Mask transform is defined, but the mask is not returned.')
