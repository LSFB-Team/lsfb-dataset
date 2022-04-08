import pytest

import tempfile
from pathlib import Path
import os
from os.path import join as path_join
import numpy as np
import pandas as pd
import cv2


@pytest.fixture(scope="session")
def lsfb_isol_path():
    """
    Create a fake directory for the dataset containing mock of videos and csv.
    """

    with tempfile.TemporaryDirectory() as tmp_dirname:

        clips_csv = create_clips_csv(tmp_dirname)
        data = pd.read_csv(clips_csv)

        os.mkdir(path_join(tmp_dirname, "videos"))
        os.mkdir(path_join(tmp_dirname, "features"))

        # Create file arborescence
        for i, row in data.iterrows():

            video_gloss_folder = path_join(tmp_dirname, "videos", row["gloss"])

            if not os.path.exists(video_gloss_folder):
                os.mkdir(video_gloss_folder)

            fake_video_path = path_join(video_gloss_folder, row["filename"])
            create_fake_video(fake_video_path)

            # Mock the csv
            features_gloss_folder = path_join(tmp_dirname, "features", row["gloss"])

            if not os.path.exists(features_gloss_folder):
                os.mkdir(features_gloss_folder)

            for feat_type in [
                "face_landmarks",
                "pose_landmarks",
                "hands_landmarks",
                "holistic_landmarks",
                "holistic_landmarks_clean",
            ]:
                feat_path = path_join(features_gloss_folder, row[f"{feat_type}"])

        yield (Path(tmp_dirname))


def create_clips_csv(base_path):
    """Create a fake clips.csv file for the LSFB isol dataset.
    The csv file is expected to contain the following columns:
    - gloss : gloss of the sign
    - hand : the hands involved in the sign (left, right or left,right).
    - relative_path : the relative path to the video file.
    - filename : the filename of the video file.
    - face_landmarks : relative path to the face landmarks csv file.
    - pose_landmarks : relative path to the pose landmarks csv file.
    - hands_landmarks : relative path to the hand landmarks csv file.
    - holistic_landmarks : relative path to the holistic landmarks csv file.
    - holistic_landmarks_clean : relative path to the holistic landmarks csv file.

    """

    config = {"GLOSS1": 3, "GLOSS2": 2, "GLOSS3": 4}
    csv_path = os.path.join(base_path, "clips.csv")
    data = []

    for gloss, nbr in config.items():
        for i in range(nbr):
            line = {}
            line["gloss"] = gloss
            line["hand"] = "left" if i % 2 == 0 else "right"

            filename = f"{gloss}_{i}.mp4"

            line["filename"] = filename
            line["relative_path"] = f"videos/{gloss}/{filename}"

            # CSV file
            line["face_landmarks"] = path_join("features", gloss, filename, "face.csv")
            line["pose_landmarks"] = path_join("features", gloss, filename, "pose.csv")
            line["hands_landmarks"] = path_join("features", gloss, filename, "hand.csv")
            line["holistic_landmarks"] = path_join(
                "features", gloss, filename, "holistic.csv"
            )
            line["holistic_landmarks_clean"] = path_join(
                "features", gloss, filename, "holistic_cleaned.csv"
            )

            data.append(line)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


def create_fake_video(video_path, len=30):
    """
    Create a fake mp4 video using OpenCV.
    """

    shape = (224, 224)

    fourc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(video_path, fourc, 30.0, shape)

    for i in range(len):
        frame = np.ones((224, 224, 3), dtype=np.uint8)
        out.write(frame)

    out.release()
