import urllib
import urllib.request
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Callable, List


class DatasetDownloader:
    """
    DatasetDownloader 

    This class provides an esay interface for downloading the LSFB dataset over http.

    Args:
        destination : Path to the destination folder for the dataset
        dataset : The dataset to download. Default = 'isol'.
            'isol' for the LSFB isol dataset;
            'cont' for the LSFB cont dataset;
        landmarks : Select which landmarks (features) to download. Default = ['pose', 'hands'].
            'pose' for pose skeleton (23 landmarks);
            'hands' for hand skeleton (21 landmarks per hand);
            'face' for face skeleton (468 landmarks).
        include_video : Download the raw video. Default = False.
        include_raw : Download the raw landmarks (festures) without smoothing and interpolation.
            Default = False.
        compute_hash : Verify the hash of the video. Include_video should be set to true to use
            this options. Default = False.
        src : URL of the source of the dataset. Default = URL of the UNamur servers
    """

    def __init__(
        self,
        destination: str,
        dataset: str = "isol",
        landmarks: List[str] = None,
        include_video: bool = False,
        include_raw: bool = False,
        compute_hash: bool = False,
        src: str = None,
    ):
        self.destination = destination
        self.dataset = dataset

        if landmarks == None:
            self.landmarks = ["pose", "hands"]
        else:
            self.landmarks = landmarks

        if include_raw:
            landmarks = self.landmarks

            for landmark in self.landmarks:
                landmarks.append(landmark + "_raw")

        self.include_video = include_video
        self.compute_hash = compute_hash

        if src != None:
            self.src = src
        elif dataset == "isol":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/LSFB_ISOL"
        elif dataset == "cont":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/LSFB_CONT"


    def download(self):
        """
        Download the data accordingly to the downloader configuration.
        """
        csv_path = self._download_csv()
        data = pd.read_csv(csv_path)

        # processing the CSV file
        for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

            if self.include_video:
                self.download_video(row)

            if self.dataset == "cont":
                self._download_annotations(row, "annots_left")
                self._download_annotations(row, "annots_right")
                self._download_annotations(row, "annots_trad")

            for landmark in self.landmarks:
                if type(row[landmark]) == str and row[landmark] != "":
                    self._download_landmarks(row[landmark])

    def _download_csv(self):
        csv_files = []

        if self.dataset == "isol":
            csv_files.append("lemmes.csv")
            csv_files.append("clips.csv")

            detailed_csv_destination = os.path.join(self.destination, "clips.csv")

        elif self.dataset == "cont":
            csv_files.append("valid_videos.csv")
            detailed_csv_destination = os.path.join(self.destination, "valid_videos.csv")

        for elem in csv_files:
            url = os.path.join(self.src, elem)
            destination = os.path.join(self.destination, elem)

            urllib.request.urlretrieve(url, destination)

        return detailed_csv_destination

    def download_video(self, row):
        """
        Download the video from the source url if the video was not yet downloaded by the client.
        This function also checks the integrity of the video already downloaded. If the md5 sum doesn't match
        the video is redownloaded.

        Input:

        row : The dataframe row containing the information for that video.
        """

        if self.dataset == 'isol':
            location = row["video"]
        else:
            location = row["filepath"]

        # Test if the video exists
        video_url = os.path.join(self.src, urllib.parse.quote(location))
        video_destination = os.path.join(self.destination, location)

        self._create_directories(video_destination)

        if not os.path.exists(video_destination):
            urllib.request.urlretrieve(video_url, video_destination)

        elif os.path.getsize(video_destination) == 0:
            urllib.request.urlretrieve(video_url, video_destination)

        elif self.dataset == "cont" and self.compute_hash:
            # Check md5sum
            dest_md5 = hashlib.md5(open(video_destination, "rb").read()).hexdigest()
            src_md5 = row["md5"]

            if dest_md5 != src_md5:
                urllib.request.urlretrieve(video_url, video_destination)

    def _download_annotations(self, row, col_name):
        """
        Download the annotations file for a video. There is one file for each hand.

        Input:
        row : The dataframe row containing the information for that video.
        hand : The hand you want to download the annotations for. Either "right_hand" or "left_hand".
        """

        location = row[col_name]

        annotation_url = os.path.join(self.src, urllib.parse.quote(location))
        annotation_destination = os.path.join(self.destination, location)

        self._create_directories(annotation_destination)

        if not os.path.exists(annotation_destination):
            urllib.request.urlretrieve(annotation_url, annotation_destination)

    def _download_landmarks(self, relative_path):
        landmarks_url = os.path.join(self.src, urllib.parse.quote(relative_path))
        landmarks_destination = os.path.join(self.destination, relative_path)

        self._create_directories(landmarks_destination)

        if not os.path.exists(landmarks_destination):
            urllib.request.urlretrieve(landmarks_url, landmarks_destination)

    def _create_directories(self, path):
        """
        Create the directories in the destination if they don't exist.

        Input:
        path : The path to the file you want to create the directories for.
        """

        directories = os.sep.join(path.split(os.sep)[:-1])

        # Creating missing directories in destination
        if not os.path.exists(directories):
            os.makedirs(directories)

