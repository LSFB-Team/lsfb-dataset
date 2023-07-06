from lsfb_dataset.datasets.lsfb_isol import (
    LSFBIsolLandmarks,
    LSFBIsolLandmarksGenerator,
    LSFBIsolBase,
)

from lsfb_dataset.datasets.types import *

"""
Builder for the LSFB isol landmarks dataset.

The builder allows to easily create a dataset class using the landmarks information for LSFB isols
while keeping the client code base readable.

Args:
    generator: Should create a generator version of the dataset. Default=False.
        The generator version will load the data on the fly instead of loading everything in memory.

Author: jfink
"""


class IsolLandmarksBuilder:
    def __init__(self, generator: bool) -> None:
        if self.generator:
            self.dataset = LSFBIsolLandmarksGenerator()
        else:
            self.dataset = LSFBIsolLandmarks()

    def reset(self) -> None:
        """
        Reset the builder at its initial state. This is useful if you want to create multiple datasets
        """
        if self.generator:
            self.dataset = LSFBIsolLandmarksGenerator()
        else:
            self.dataset = LSFBIsolLandmarks()

    def set_root(self, root: str):
        """
        Set the root path for the ISOL dataset on your system. The dataset should already be downloaded.

        Args:
            root: Path to the root directory of the dataset.
        """
        self.dataset.config.root = root
        self.dataset.config.__post_init__()

    def set_landmarks(self, landmarks: list[str]):
        """
        Set the landmarks to use for the dataset.

        Args:
            landmarks: List of landmarks to use. Default = ['pose', 'hand_left', 'hand_right'].
        """
        self.dataset.config.landmarks = landmarks
        self.dataset.config.__post_init__()

    def set_features_transforms(self, transform: callable):
        """
        Set a callable object (e.g., function) for landmarks pre-processing. The callable will received features as a numpy array
        containing all the landmarks for all the frames of the video and should return the landmarks preprocessed.

        Args:
            transform: A callable object to pre-process landmarks.

        """
        self.dataset.config.features_transform = transform

    def set_target_transform(self, transform: callable):
        """
        Set a callable object (e.g., function) for target pre-processing. The callable will receive the target and should return
        the preprocessed target

        Args:
            transform: A callable object to pre-process the target.
        """

        self.dataset.config.target_transform = transform

    def set_transform(self, transform: callable) -> None:
        """
        Set a callable object (e.g., function) to pre-process the features and the target. The callable will receive
        the features as a numpy array along with the target and should return the pre-processed features and target.

        Args:
            transform: A callable object to pre-process the features and target.
        """
        self.dataset.config.transform = transform

    def set_mask_transform(self, transform: callable) -> None:
        """
        Set a callable object (e.g., function) to pre-process the mask information. The callable will receive a mask
        and should return the pre-processed mask. The use_mask attribute should be set to True.

        Args:
            transform: A callable object to pre-process the mask
        """
        self.dataset.config.mask_transform = transform

    def set_instance_nb(self, number: int) -> None:
        """
        Set the number of lemmes that will be used by the dataset. The lemmes selected are the n first lemmes appearing
        in the lemmes.csv file.

        Args:
            number: Number of lemmes to use.
        """
        self.dataset.config.instance_nb = number

    def set_instances_list_file(self, filename: str) -> None:
        """
        Set the path to the lemmes list csv file.

        Args:
            filename: Path to the lemmes list file.
        """
        self.dataset.config.instances_list_file = filename
        self.dataset.config.__post_init__()

    def set_split(self, splitname: DataSubset) -> None:
        """
        Set the data split to use for the dataset. The available splits are ['all', 'train', 'test', 'mini_sample'].
        Use mini_sample for debugging purposes.

        Args:
            splitname: Name of the split to use.
        """
        self.dataset.config.split = splitname
        self.dataset._select_instances()

    def set_sequence_max_length(self, length: int) -> None:
        """
        Set the maximum length of the sequences. Sequences longer than this value will be truncated.

        Args:
            length: Maximum length of the sequences.
        """
        self.dataset.config.sequence_max_length = length

    def set_padding(self, padding: bool) -> None:
        """
        Set the padding option. If set to True, the sequences will be padded to the maximum length.

        Args:
            padding: True if the sequences should be padded to the maximum length.
        """
        self.dataset.config.padding = padding

    def set_use_mask(self, use_mask: bool) -> None:
        """
        Set the use_mask option. If set to True, the dataset will return a mask along with the features and the target.
        The mask indicate which frames are padded.

        Args:
            use_mask: True if the mask should be used.
        """
        self.dataset.config.use_mask = use_mask

    def set_mask_value(self, value: int) -> None:
        """
        Set the value to use to indicate that the frame are padded.

        Args:
            value: Value to use for the mask.
        """
        self.dataset.config.mask_value = value

    def set_show_progress(self, show_progress: bool) -> None:
        """
        Set the show_progress option. If set to True, the dataset will show a progress bar when loading the data.

        Args:
            show_progress: True if a progress bar should be shown.
        """
        self.dataset.config.show_progress = show_progress

    def get_result(self) -> LSFBIsolBase:
        """
        Return the dataset created by the builder.

        Returns:
            The dataset created by the builder.
        """
        return self.dataset
