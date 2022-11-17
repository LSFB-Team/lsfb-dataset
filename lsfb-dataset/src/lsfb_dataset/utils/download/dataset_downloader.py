from urllib.parse import quote, urljoin
from pathlib import PurePath
import requests
import hashlib
import pandas as pd
from tqdm import tqdm
import os
from os import path
from typing import List
from ..datasets import split_cont, split_isol, mini_sample


class DatasetDownloader:
    """
    DatasetDownloader 

    This class provides an easy interface for downloading the LSFB dataset over http.

    Args:
        destination : Path to the destination folder where the dataset is saved.
        dataset : The dataset to download. Default = 'isol'.
            'isol' for the LSFB isol dataset;
            'cont' for the LSFB cont dataset;
        landmarks : Select which landmarks (features) to download. Default = ['pose', 'hands'].
            'pose' for pose skeleton (23 landmarks);
            'hands' for hand skeleton (21 landmarks per hand);
            'face' for face skeleton (468 landmarks).
        split : Select which subset of the dataset should be downloaded. Default = 'all'.
            'all' for the entire dataset;
            'train' for the training data only;
            'test' for the test data only;
            'mini_sample' for a tiny sample of the dataset.
        include_video : Download the raw video. Default = False.
        include_raw : Download the raw landmarks (features) without smoothing nor interpolation.
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
            split: str = 'all',
            include_video: bool = False,
            include_raw: bool = False,
            compute_hash: bool = False,
            src: str = None,
    ):
        self.destination = destination
        self.dataset = dataset
        self.split = split

        if landmarks is None:
            self.landmarks = ["pose", "hands"]
        else:
            self.landmarks = landmarks

        if include_raw:
            landmarks = [*self.landmarks]
            for landmark in landmarks:
                self.landmarks.append(landmark + "_raw")

        self.include_video = include_video
        self.compute_hash = compute_hash

        if src is not None:
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
        data = self._select_split(data)

        progress_bar = tqdm(data.iterrows(), total=data.shape[0])
        for idx, row in progress_bar:
            progress_bar.set_postfix_str(row['filename'])

            if self.include_video:
                self.download_video(row)

            if self.dataset == "cont":
                self._download_annotations(row, "annots_left")
                self._download_annotations(row, "annots_right")
                self._download_annotations(row, "annots_trad")
                self._download_segmentation_vectors()

            for landmark in self.landmarks:
                if type(row[landmark]) == str and row[landmark] != "":
                    self._download_landmarks(row[landmark])

    def _download_csv(self):
        csv_files = []

        if self.dataset == "isol":
            csv_files.append("lemmes.csv")
            csv_files.append("clips.csv")
            metadata_dest = path.join(self.destination, "clips.csv")

        elif self.dataset == "cont":
            csv_files.append("videos.csv")
            metadata_dest = path.join(self.destination, "videos.csv")

        else:
            raise ValueError(f'Unknown dataset: {self.dataset}')

        for origin_path in csv_files:
            destination = os.path.join(self.destination, origin_path)
            self._download_file(origin_path, destination)

        return metadata_dest

    def download_video(self, row):
        """
        Download the video from the source url if the video was not yet downloaded by the client.
        This function also checks the integrity of the video already downloaded. If the md5 sum doesn't match
        the video is re-downloaded.

        Input:

        row : The dataframe row containing the information for that video.
        """

        if self.dataset == 'isol':
            video_path = row["video"]
        else:
            video_path = row["filepath"]

        video_destination = os.path.join(self.destination, video_path)

        if not os.path.isfile(video_destination):
            self._download_file(video_path, video_destination)

        elif os.path.getsize(video_destination) == 0:
            self._download_file(video_path, video_destination)

        elif self.dataset == "cont" and self.compute_hash:
            # Check md5sum
            dest_md5 = hashlib.md5(open(video_destination, "rb").read()).hexdigest()
            src_md5 = row["md5"]

            if dest_md5 != src_md5:
                self._download_file(video_path, video_destination)

    def _download_annotations(self, row, col_name):
        """
        Download the annotations file for a video.
        There is one file for each hand.

        Input:
        row : The dataframe row containing the information for that video.
        hand : The hand you want to download the annotations for. Either "right_hand" or "left_hand".
        """

        annot_path = row[col_name]
        annotation_destination = os.path.join(self.destination, annot_path)
        if not os.path.exists(annotation_destination):
            self._download_file(annot_path, annotation_destination)

    def _download_segmentation_vectors(self):
        destinations = [
            (f'annotations/vectors/{x}.pck', path.join(self.destination, 'annotations/vectors', f'{x}.pck'))
            for x in ['activity', 'binary', 'binary_with_coarticulation']
        ]
        for origin, dest in destinations:
            self._download_file(origin, dest)

    def _download_landmarks(self, landmark_path):
        landmarks_destination = os.path.join(self.destination, landmark_path)
        if not os.path.exists(landmarks_destination):
            self._download_file(landmark_path, landmarks_destination)

    def _download_file(self, origin: str, destination: str):
        url = urljoin(f'{self.src}/', quote(origin))

        req = requests.get(url, allow_redirects=False, timeout=60)
        if req.status_code != 200:
            raise FileNotFoundError(f"""
            Could not download the desired file: {origin}.
            Wrong  HTTP status ({req.status_code}) for request ({url}).
            """)

        os.makedirs(str(PurePath(destination).parent), exist_ok=True)
        with open(destination, 'wb') as file:
            file.write(req.content)

    def _select_split(self, data):
        if self.split == 'all':
            return data

        if self.split == 'mini_sample':
            return mini_sample(data)

        if self.dataset == 'isol':
            train, test = split_isol(data)
        else:
            train, test = split_cont(data)

        if self.split == 'train':
            return train
        elif self.split == 'test':
            return test
        else:
            raise ValueError(f'Unknown split: {self.split}')
