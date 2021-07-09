import pytest

import tempfile
from pathlib import Path
import os
import numpy as np
import cv2


@pytest.fixture
def lsfb_isol_path():

    with tempfile.TemporaryDirectory() as tmp_dirname:

        path = Path(tmp_dirname)

        for sign in ["mock1", "mock2"]:
            sign_folder_path = f"{path}{os.sep}{sign}"
            os.mkdir(sign_folder_path)

            for split in ["train", "test", "rejected"]:
                split_folder = f"{sign_folder_path}{os.sep}{split}"

                os.mkdir(split_folder)

                for i in range(4):
                    # Create fake mp4
                    vid_path = f"{split_folder}{os.sep}{split}_{i}.avi"
                    create_fake_video(vid_path)

        yield (Path(tmp_dirname))


def create_fake_video(video_path):

    shape = (224, 224)

    fourc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(video_path, fourc, 30.0, shape)

    for i in range(30):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        out.write(frame)

    out.release()

