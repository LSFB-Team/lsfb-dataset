from lsfb_dataset.utils.datasets import split_isol, mini_sample
from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
import abc

import pandas as pd


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
        config: The configuration object (see LSFBContConfig).
            If config is not specified, every needed configuration argument must be manually provided.

    Configuration args (see LSFBIsolConfig):
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
            'mini_sample' for a tiny set of instances;
        sequence_max_length: Max length of the clip sequence. Default=50.
        padding: Pad all sequence to the same length.
        return_mask: Returning padding mask for the sequence.
        mask_value: Value of the masked part of the clips.
        show_progress: If true, show a progress bar while the dataset is loading.

    """

    def __init__(self, config=None, **kwargs):
        self.config = LSFBIsolConfig(**kwargs) if config is None else config
        self.config.instances = self._select_instances()

    def _select_instances(self):
        split = self.config.split
        self._filter_instances_list()

        instances = self.config.instances

        if split == "mini_sample":
            instances = mini_sample(instances)
        elif split != "all":
            train_instances, test_instances = split_isol(instances)
            if split == "train":
                instances = train_instances
            elif split == "test":
                instances = test_instances

        return instances

    def _filter_instances_list(self):
        # unique values pandas dataframe
        unique_signs = self.config.instances["sign"].unique()
        retained_sign = []

        for sign in unique_signs:
            if (
                len(self.config.instances[self.config.instances["sign"] == sign])
                >= self.config.instance_nb
            ):
                retained_sign.append(sign)

        # filter instances list
        self.config.instances.drop(
            index=self.config.instances.index[
                ~self.config.instances["sign"].isin(retained_sign)
            ],
            inplace=True,
        )

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass
