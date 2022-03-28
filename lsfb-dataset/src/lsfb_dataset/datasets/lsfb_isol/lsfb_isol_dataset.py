from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, Callable, List
import cv2
import pandas as pd
import numpy as np
import os


class LsfbIsolDataset(Dataset):

    """
    Load the LSFB_ISOL dataset.
    The dataset contains video sequence of size 720x576 recorded in 50FPS.

    Those video sequences could be enriched with the following features :
      - Hands landmarks : Media Pipe landamarks for the hands of the signer
      - Face landmarks : Media Pipe landmarks for the face of the signer
      - Pose landmarks : Media Pipe landmarks for the pose of the signer

    Target values :
      - The gloss associated with the video

    """

    def __init__(
        self,
        root: str,
        transforms: Optional[List[Callable]] = None,
        features: Optional[List[str]] = None,
        labels: Optional[Dict[int, str]] = None,
        max_frame: Optional[int] = 50,
    ):
        """
        Load the dataset.

        Parameters
        ----------
        root : str
            The root folder containing the dataset
        transform : Optional[Callable]
            The pytorch transform
        features : list[str]
            List of the feature to load
        max_frame:
            The maximum number of frames to load. Use it to save memory.
        """

        self.root: str = root
        self.transforms = transforms
        self.features = features
        self.labels = labels
        self.max_frame = max_frame

        self.clips_info = pd.read_csv(os.path.join(root, "clips.csv"))
        self.clips_info = self._filter_clips_info(self.clips_info, features)

        if self.features == None:
            self.features = ["video"]

        if self.labels == None:
            self.labels = self._load_label_mapping()

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.clips_info)

    def __getitem__(self, idx: int):
        """
        Return an item of the dataset.
        """
        X = {}

        item = self.clips_info.iloc[idx]
        y = self.labels[item["gloss"]]

        if "video" in self.features:
            vid_path = os.path.join(self.root, item["relative_path"])
            X["video"] = self._load_video(vid_path)

        if "hands_landmarks" in self.features:
            landmarks_path = os.path.join(self.root, item["hands_landmarks"])
            X["hands_landmarks"] = pd.read_csv(landmarks_path).to_numpy()

        if "face_landmarks" in self.features:
            landmarks_path = os.path.join(self.root, item["face_landmarks"])
            X["face_landmarks"] = pd.read_csv(landmarks_path).to_numpy()

        if "pose_landmarks" in self.features:
            landmarks_path = os.path.join(self.root, item["pose_landmarks"])
            X["pose_landmarks"] = pd.read_csv(landmarks_path).to_numpy()

        # Applying the transform
        if self.transforms:
            for transform in self.transforms:
                X = transform(X)

        return X, y

    def _load_video(self, path: str):
        """
        Load all
        """
        capture = cv2.VideoCapture(path)

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
            # video if it is > max_frame (default 150 ≃ 5sec)
            if frame_count > self.max_frame:
                break

        return np.array(frame_array)

    def _load_label_mapping(self) -> Dict[int, str]:
        """
        Create a mapping between the gloss string and an integer value.
        Returns:

        label_mapping : dictionnary of int : str containing the mapping.

        """
        unique_gloss = self.clips_info["gloss"].unique().tolist()
        label_mapping = dict(map(lambda x: (x, unique_gloss.index(x)), unique_gloss))
        return label_mapping

    def _filter_clips_info(
        self, clips_info: pd.DataFrame, feature: list[str]
    ) -> pd.DataFrame:
        for feature in feature:
            if feature == "video":
                clips_info = clips_info[~clips_info["relative_path"].isnull()]

            else:
                clips_info = clips_info[~clips_info[feature].isnull()]

        return clips_info
