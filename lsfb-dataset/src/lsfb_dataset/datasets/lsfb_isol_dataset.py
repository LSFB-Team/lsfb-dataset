from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import cv2
import numpy as np
import random


class LsfbIsolDataset(Dataset):
    """Load the LSFB video based on a dataframe containing their path"""

    def __init__(
        self,
        data,
        label_padding="loop",
        sequence_label=False,
        transforms=None,
        max_frame=150,
        labels=None,
    ):
        """
        data : A pandas dataframe containing ...
        nbr_frames : Number of frames to sample per video
        label_padding : Required when using sequence label. Indicates if the padding should be a specific padding value
                        or if the padding should be looped.
        sequence_label : Return one label per video frame
        transforms : transformations to apply to the frames.
        max_frame : Maximum number of frames to load for each video
        """
        self.data = data
        self.label_padding = label_padding
        self.sequence_label = sequence_label
        self.transforms = transforms
        self.max_frame = max_frame

        if labels == None:
            self.labels = self._get_label_mapping()
        else:
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]

        capture = cv2.VideoCapture(item["path"])
        video = self.extract_frames(capture)
        video_len = len(video)

        # Apply the transformations to the video :
        if self.transforms:
            video = self.transforms(video)

        y = int(item["label_nbr"])

        ## Handle the sequence label
        if self.sequence_label and self.label_padding == "loop":
            y = np.array([y] * len(video))
        elif self.sequence_label:
            pad_size = len(video) - video_len

            if pad_size > 0:
                pad_label = list(self.labels.keys())[
                    list(self.labels.values()).index("<pad>")
                ][0]
                y = np.array([y] * video_len) + np.array([pad_label] * pad_size)
            else:
                y = np.array([y] * len(video))

        return video, y

    def _get_label_mapping(self) -> Dict[int, str]:
        labels = self.data.label.unique()

        mapping = {}
        for label in labels:

            subset = self.data[self.data["label"] == label]
            class_number = subset["label_nbr"].iloc[0]

            mapping[class_number] = label

        if self.label_padding == "zero" and self.sequence_label:
            nbr_class = len(mapping)
            mapping[nbr_class] = "<pad>"

        return mapping

    def extract_frames(self, capture: cv2.VideoCapture):

        frame_array = []
        success, frame = capture.read()

        # Select
        frame_count = 0
        while success:
            frame_count += 1

            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            frame_array.append(frame / 255)
            success, frame = capture.read()

            # Avoid memory saturation by stopping reading of
            # video if it is > 150 frames (5sec)
            if frame_count > self.max_frame:
                break

        return np.array(frame_array)
