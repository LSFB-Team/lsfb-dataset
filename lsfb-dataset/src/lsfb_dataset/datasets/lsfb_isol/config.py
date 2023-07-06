from dataclasses import dataclass
from typing import Callable, List, Optional
from lsfb_dataset.datasets.types import *
import os
import pandas as pd


@dataclass
class LSFBIsolConfig:
    """
    Simple configuration class for the lsfb_isol Dataset.

    lsfb: French Belgian Sign Language
    isol: isolated videos in LSFB

    Args:
        root: Root directory of the LSFB_ISOL dataset.
            The dataset must already be downloaded.

        landmarks: Select which landmarks (features) to use. Default = ['pose', 'hand_left', 'hand_right'].
            'pose' for pose skeleton (23 landmarks);
            'hands_left' for left hand skeleton (21 landmarks);
            'hands_right' for right hand skeleton (21 landmarks);

        features_transform: Callable object used to transform the features.
        target_transform: Callable object used to transform the targets.
        transform: Callable object used to transform both the features and the targets.

        mask_transform: Callable object used to transform the masks.
            You need to set return_mask to true to use this transform.

        instance_nb: Minimum number of instances per label. Default=10
        instances_list_file: Path to the csv containing the lemmes lists. Default="instances.csv"

        split: Select a specific subset of the dataset. Default = 'all'.
            'train' for training set;
            'test' for the test set;
            'all' for all the instances of the dataset;
            'mini_sample' for a tiny set of instances.

        sequence_max_length: Max length of the clip sequence. Default=50.
        padding: Pad all sequence to the same length.
        return_mask: Returning padding mask for the sequence.
        mask_value: Value of the masked part of the clips.
        show_progress: If true, show a progress bar while the dataset is loading.


    Author: jfink
    """

    root: str
    landmarks: Optional[List[str]] = None
    use_3d: bool = False

    features_transform: Callable = None
    target_transform: Callable = None
    transform: Callable = None
    mask_transform: Callable = None

    instance_nb: int = 10
    instances_list_file: str = "instances.csv"

    split: DataSubset = "all"
    sequence_max_length: int = 50

    padding: bool = False
    return_mask: bool = False
    mask_value: int = 0
    show_progress: bool = True

    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = ["pose", "hand_left", "hand_right"]

        self.instances_list_path = os.path.join(self.root, self.instances_list_file)
        self.instances = pd.read_csv(self.instances_list_path)

