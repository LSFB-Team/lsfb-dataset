import urllib
import hashlib
import pandas as pd
from tqdm import tqdm
import os


class DatasetDownloader:
    """
    This dataloader allows you to retrieve and sync the LSFB_CONT dataset.

    """

    def __init__(
        self,
        destination,
        dataset="isol",
        exclude=None,
        src=None,
    ):

        self.destination = destination
        self.dataset = dataset
        self.exclude = exclude

        if src != None:
            self.src = src
        elif dataset == "isol":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/lsfb_isol"
        elif dataset == "cont":
            self.src = "https://lsfb.info.unamur.be/static/datasets/LSFB/lsfb_cont"

    def download_and_sync(self):
        """
        Orchestrate the download and sync of the LSFB_CONT dataset.
        """
        csv_path = self.download_csv()
        data = pd.read_csv(csv_path)

        # processing the CSV file
        for idx, row in tqdm(data.iterrows()):

            if "video" not in self.exclude:
                self.download_video(row)

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

    def _create_directories(self, path):
        """
        Create the directories in the destination if they don't exist.
        """

        directories = os.sep.join(path.split(os.sep)[:-1])

        # Creating missing directories in destination
        if not os.path.exists(directories):
            os.makedirs(directories)
