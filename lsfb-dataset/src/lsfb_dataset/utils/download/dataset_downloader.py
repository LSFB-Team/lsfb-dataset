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
    DatasetDownloader is an easy to use interface for downloading the LSFB datasets. The download is done via http.
    """

    def __init__(
        self,
        destination: str,
        dataset: str = "isol",
        exclude: List[str] = [],
        src: str = None,
        warning_message: bool = True,
    ):
        """
        The constructor initialize the setting you want for the download.

        INPUT:
        - destination: the destination folder where the dataset will be downloaded.
        - dataset: the dataset you want to download. either "isol" or "cont". (default: "isol")
        - exclude: a list of strings containing the element you want to exclude from the download. The excluded element could be video, annotations or landmarks (default: [])
        - src: the source url of the dataset. By default, the dataset is downloaded from the unamur server located here : lsfb.info.unamur.be
        - warning_message: if True, a warning message will be displayed to warn you about the size of the download. (default: True)
        """

        self.destination = destination
        self.dataset = dataset
        self.exclude = exclude
        self.warning_message = warning_message

        if src != None:
            self.src = src
        elif dataset == "isol":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/lsfb_isol"
        elif dataset == "cont":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/lsfb_cont"

    def download(self):
        """
        The main function orchestrating all the download. Call it to start downloading data.
        """
        csv_path = self.download_csv()
        data = pd.read_csv(csv_path)

        if not self._display_warning():
            return

        # processing the CSV file
        for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

            if "video" not in self.exclude:
                self.download_video(row)

            if self.dataset == "cont" and "annotations" not in self.exclude:
                self.download_annotations(row, "right_hand")
                self.download_annotations(row, "left_hand")

            if "landmarks" not in self.exclude:

                available_landmarks = [
                    "face_landmarks",
                    "pose_landmarks",
                    "hands_landmarks",
                    "holistic_landmarks",
                    "holistic_landmarks_clean",
                ]

                for landmark in available_landmarks:
                    if type(row[landmark]) == str and row[landmark] != "":
                        self.download_landmarks(row[landmark])

    def download_csv(self):

        if self.dataset == "isol":
            csv_path = "clips.csv"
        elif self.dataset == "cont":
            csv_path = "videos.csv"

        url = os.path.join(self.src, csv_path)
        destination = os.path.join(self.destination, csv_path)

        urllib.request.urlretrieve(url, destination)

        return destination

    def download_video(self, row):
        """
        Download the video from the source url if the video was not yet downloaded by the client.
        This function also checks the integrity of the video already downloaded. If the md5 sum doesn't match
        the video is redownloaded.

        Input:

        row : The dataframe row containing the information for that video.
        """
        location = row["relative_path"]

        # Test if the video exists
        video_url = os.path.join(self.src, urllib.parse.quote(location))
        video_destination = os.path.join(self.destination, location)

        self._create_directories(video_destination)

        if not os.path.exists(video_destination):
            urllib.request.urlretrieve(video_url, video_destination)

        elif os.path.getsize(video_destination) == 0:
            urllib.request.urlretrieve(video_url, video_destination)

        elif self.dataset == "cont":
            # Check md5sum
            dest_md5 = hashlib.md5(open(video_destination, "rb").read()).hexdigest()
            src_md5 = row["md5"]

            if dest_md5 != src_md5:
                urllib.request.urlretrieve(video_url, video_destination)

    def download_annotations(self, row, hand):
        """
        Download the annotations file for a video. There is one file for each hand.

        Input:
        row : The dataframe row containing the information for that video.
        hand : The hand you want to download the annotations for. Either "right_hand" or "left_hand".
        """

        col = hand + "_annotations"
        location = row[col]

        annotation_url = os.path.join(self.src, urllib.parse.quote(location))
        annotation_destination = os.path.join(self.destination, location)

        self._create_directories(annotation_destination)

        if not os.path.exists(annotation_destination):
            urllib.request.urlretrieve(annotation_url, annotation_destination)

    def download_landmarks(self, relative_path):
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

    def _ask_permission(self, message):
        """
        Ask the user for permission to download the file.

        Input:
        message : The message to display to the user.
        """

        print(message)
        answer = input("Do you want to continue ? [Y/n] ")

        if answer.lower() == "n":
            raise Exception("User cancelled the download")

    def _display_warning(self) -> bool:
        """
        Display a warning message to the user.
        """
        if not self.warning_message:
            return True

        if self.dataset == "isol":
            size = "~100 GB"
        else:
            size = "~1 TB"

        message = f"The dataset you are downloading is {size}."
        message += (
            f" Are you sure you want to download it here : {self.destination} (Y/N): "
        )

        response = input(message)

        while response.lower() not in ["y", "n"]:
            response = input("Answer should be Y or N :  ")

        return response.lower() == "y"
