from typing import Optional
import urllib.parse as url_parse
import requests
import os
import pathlib
import json

from tqdm.contrib.concurrent import thread_map


class Downloader:
    """
    Downloader

    This class provides an easy interface for downloading the LSFB datasets over http.
    It is useful for selecting parts of the dataset you want to download, as LSFB datasets are heavy.

    Be sure to save some space for the dataset on the disk!

    Example:
        ```python
        my_downloader = Downloader(
            './my_dest_folder',
            'isol',
            splits=['fold_1', 'fold_2'],
            include_videos=True,
            include_cleaned_poses=True,
            skip_existing_files=False,
        )
        my_downloader.download()
        ```

    Args:
        destination: Path to the destination folder where the dataset is saved.
        dataset: The dataset to download. Default = 'isol'.
            'isol' for the LSFB isol dataset,
            'cont' for the LSFB cont dataset;

        splits: Select which subsets of the dataset should be downloaded. Default = ['all'].
            'all' for the entire dataset,
            'fold_0' to 'fold_4' for specific folds of the dataset,
            'train' for the training data only,
            'test' for the test data only,
            'mini_sample' for a tiny sample of the dataset.
        signers: (Optional) Specify the ids of the only signers for which videos are downloaded. Default = None.
            For example: [1, 43, 17]
        landmarks: Select which landmarks (pose features) to download.
            Default = ['face', 'pose', 'left_hand', 'right_hand'].
            'pose' for body skeleton (23 landmarks);
            'left_hand' or 'right_hand' for hands skeleton (21 landmarks per hand);
            'face' for face skeleton (468 landmarks).

        include_videos: Download the raw video. Default = False.
            Be careful, video data are heavy!
        include_cleaned_poses: Download the cleaned poses with interpolation and smoothing. Default = True.
        include_raw_poses: Download the raw poses without smoothing nor interpolation. Default = False.

        skip_existing_files: If true, does not download files that already exist in the destination folder.
            Otherwise, re-download the entire dataset. Default = True.

        max_parallel_connections: The number of maximum parallel HTTP connections to download the files. Default = 10.
        source: (Optional) Replace the original source (URL) of the dataset by your own. Default = None.
            By default, the downloader uses the URLs of the UNamur servers.
        check_ssl: If true, enforce the use of HTTPS (SSL/TLS) connections. Default = True.
        timeout: Number of seconds before timeout. Default = 60.
    """

    def __init__(
            self,
            dataset: str,
            destination: str,

            # Instance filtering
            splits: Optional[list[str]] = None,
            signers: Optional[list[int]] = None,
            landmarks: Optional[list[str]] = None,

            # File filtering
            include_videos: bool = False,
            include_cleaned_poses: bool = True,
            include_raw_poses: bool = False,

            # Pre- and post-processing
            skip_existing_files: bool = True,
            # check_file_hash: bool = False, # TODO

            # Other
            max_parallel_connections: int = 10,
            source: Optional[str] = None,
            check_ssl: bool = True,
            timeout: int = 60,
    ):
        self.dataset = dataset.lower()

        self.splits = splits
        if splits is None:
            self.splits = ['all']
        self.signers = signers

        self.landmarks = landmarks
        if landmarks is None:
            self.landmarks = ['pose', 'left_hand', 'right_hand', 'face']

        self.include_videos = include_videos
        self.include_raw_poses = include_raw_poses
        self.include_cleaned_poses = include_cleaned_poses

        if source is not None:
            self.source = source
        elif dataset == 'isol':
            self.source = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
        elif dataset == 'cont':
            self.source = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/cont"
        else:
            raise ValueError("Unknown dataset.")

        self.destination = pathlib.Path(destination)
        if not os.path.isdir(self.destination):
            raise FileNotFoundError(f'Destination directory not found: {self.destination}.')

        self.skip_existing_file = skip_existing_files
        self.max_parallel_connections = max_parallel_connections
        self.check_ssl = check_ssl
        self.timeout = timeout

        self.instances = []

    def _get_metadata_origins(self):
        metadata = []
        for split in ['all', 'mini_sample', 'test', 'train'] + [f'fold_{k}' for k in range(5)]:
            metadata.append(f'metadata/splits/{split}.json')
        metadata.append('metadata/sign_occurrences.csv')
        metadata.append('metadata/sign_to_index.csv')
        metadata.append('instances.csv')
        if self.dataset == 'isol':
            metadata.append('instances_special.csv')
        return metadata

    def _get_cont_annotations_origins(self):
        annotations = []
        for prefix, suffix in zip(['signs', 'special_signs'], ['both_hands', 'left_hand', 'right_hand']):
            annotations.append(f'annotations/{prefix}_{suffix}.json')
        annotations.append('annotations/subtitles.json')
        return annotations

    def _get_pose_origins(self, raw: bool = False):
        poses = []
        pose_folder = 'poses_raw' if raw else 'poses'
        for instance_id in self.instances:
            for pose_type in self.landmarks:
                poses.append(f'{pose_folder}/{pose_type}/{instance_id}.npy')
        return poses

    def _get_video_origins(self):
        videos = []
        for instance_id in self.instances:
            videos.append(f'videos/{instance_id}.npy')
        return videos

    def _get_instances(self):
        self.instances = []
        for split in self.splits:
            with open(f'{self.destination}/metadata/splits/{split}.json') as file:
                self.instances += json.load(file)
        self.instances = list(set(self.instances))
        if self.signers is not None:
            self.instances = [instance for instance in self.instances if int(instance[13:16]) in self.signers]

    def _download_file(self, origin: str):
        filepath = self.destination / origin
        if self.skip_existing_file and os.path.isfile(filepath):
            return

        url = url_parse.urljoin(f"{self.source}/", url_parse.quote(origin))
        req = requests.get(
            url, allow_redirects=False, timeout=self.timeout, verify=self.check_ssl
        )

        if req.status_code != 200:
            raise FileNotFoundError(
                f"""
                Could not download the desired file: {origin}.
                Wrong  HTTP status ({req.status_code}) for request ({url}).
                """
            )

        # The server responds an empty HTML page if not found...
        if req.status_code == 200 and req.headers.get("Content-Type") == "text/html":
            raise FileNotFoundError(
                f"""
                Could not download the desired file: {origin}.
                Resource not found ({url}).
                """
            )

        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(req.content)

    def _download_files(self, origins: list[str], title: Optional[str] = None, unit: str = 'files'):
        thread_map(
            self._download_file, origins,
            max_workers=self.max_parallel_connections,
            unit=unit,
            desc=title,
        )

    def download(self):
        self._download_files(self._get_metadata_origins(), title='Metadata')
        if self.dataset == 'cont':
            self._download_files(self._get_cont_annotations_origins(), title='Annotations')
        self._get_instances()
        if self.include_cleaned_poses:
            self._download_files(
                self._get_pose_origins(),
                title=f'Poses for {len(self.instances)} instances [{", ".join(self.landmarks)}]'
            )
        if self.include_raw_poses:
            self._download_files(
                self._get_pose_origins(raw=True),
                title=f'Raw poses for {len(self.instances)} instances [{", ".join(self.landmarks)}]'
            )
        if self.include_videos:
            self._download_files(
                self._get_video_origins(),
                title=f'Videos for {len(self.instances)} instances'
            )


if __name__ == "__main__":
    downloader = Downloader(
        'isol',
        '/run/media/ppoitier/ppoitier/datasets/sign-languages/test_lsfb',
        skip_existing_files=False,
        splits=['mini_sample'],
        include_videos=True,
        include_raw_poses=False,
        include_cleaned_poses=True,
    )
    downloader.download()
