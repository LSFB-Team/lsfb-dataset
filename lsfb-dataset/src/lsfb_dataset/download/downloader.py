from typing import Optional
import urllib.parse as url_parse
import requests
import os
import pathlib

from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm


class Downloader:
    def __init__(
            self,
            dataset: str,
            destination: str,

            # Instance filtering
            splits: Optional[list[str]] = None,
            signers: Optional[list[str]] = None,
            metadata: Optional[list[str]] = None,

            # File filtering
            include_videos: bool = False,
            include_raw_landmarks: bool = False,
            include_cleaned_landmarks: bool = True,

            # Pre- and post-processing
            skip_existing_files: bool = True,
            check_file_hash: bool = False,

            # Other
            max_parallel_connections: int = 10,
            source: Optional[str] = None,
            check_ssl: bool = True,
    ):
        if source is not None:
            self.source = source
        elif dataset == 'isol':
            self.source = "https://lsfb.info.unamur.be/static/datasets/LSFB/LSFB_ISOL"
        elif dataset == 'cont':
            self.source = "https://lsfb.info.unamur.be/static/datasets/LSFB/LSFB_CONT"
        else:
            raise ValueError("Unknown dataset.")

        self.destination = pathlib.Path(destination)
        if not os.path.isdir(self.destination):
            raise FileNotFoundError(f'Destination directory not found: {self.destination}.')

        self.skip_existing_file = skip_existing_files
        self.max_parallel_connections = max_parallel_connections
        self.check_ssl = check_ssl


    def _get_video_origins(self):
        pass

    def _get_landmarks_origins(self):
        pass

    def _download_file(self, origin: str):
        filepath = self.destination / origin
        if self.skip_existing_file and os.path.isfile(filepath):
            return

        url = url_parse.urljoin(f"{self.source}/", url_parse.quote(origin))
        req = requests.get(
            url, allow_redirects=False, timeout=60, verify=self.check_ssl
        )

        # The server responds an empty HTML page if not found...
        if req.status_code != 200 or req.headers["Content-Type"] == "text/html":
            raise FileNotFoundError(
                f"""
                Could not download the desired file: {origin}.
                Wrong  HTTP status ({req.status_code}) for request ({url}).
                """
            )

        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(req.content)

    def _download_files(self, origins: list[str]):
        thread_map(
            self._download_file, origins,
            max_workers=self.max_parallel_connections,
            unit='videos',
        )
        # Parallel(n_jobs=self.max_parallel_connections)(delayed(self._download_file)(origin) for origin in origins)


if __name__ == "__main__":
    downloader = Downloader('cont', '/run/media/ppoitier/ppoitier/datasets/sign-languages/lsfb/cont', skip_existing_files=False)
    downloader._download_files(['videos.csv', 'videos_unchecked.csv'])
