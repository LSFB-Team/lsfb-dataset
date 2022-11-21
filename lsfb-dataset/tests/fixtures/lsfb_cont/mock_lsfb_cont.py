import pytest

import tempfile
import pandas as pd
from pathlib import Path
from fixtures.landmarks import hands_landmarks_list, pose_landmarks_list
import random
import os
from datetime import datetime
import numpy as np
import hashlib
from pathlib import PurePath


@pytest.fixture(scope="session")
def mock_lsfb_cont_path():
    """
    Mock a lsfb isol dataset containing 10 signs label with 30 examples each.
    """
    with tempfile.TemporaryDirectory() as tmp_dirname:

        videos = create_dummy_videos_csv(tmp_dirname)

        for _, row in videos.iterrows():

            frames = row["frames"]

            pose_raw = row["pose_raw"]
            pose = row["pose"]
            hands_raw = row["hands_raw"]
            hands = row["hands"]

            create_landmarks_csv(tmp_dirname, pose_raw, hands_raw, hands, pose, frames)

            # Creating annotations

            left_hand_csv = row["annots_left"]
            right_hand_csv = row["annots_right"]
            translation_csv = row["annots_trad"]
            millisec = row["duration"]

            create_annotations_csv(
                tmp_dirname, left_hand_csv, right_hand_csv, translation_csv, millisec
            )

        yield tmp_dirname


def create_dummy_videos_csv(
    tmp_path: str, nb_session: int = 4, nb_records: int = 2, nb_signers=10
):

    data = {
        "eaf": [],
        "session": [],
        "record": [],
        "filename": [],
        "filepath": [],
        "signer": [],
        "found": [],
        "hash": [],
        "datetime": [],
        "fps": [],
        "frames": [],
        "duration": [],
        "width": [],
        "height": [],
        "annots_left": [],
        "annots_right": [],
        "annots_trad": [],
        "pose_raw": [],
        "hands_raw": [],
        "face_raw": [],
        "pose": [],
        "hands": [],
        "face": [],
    }

    signers_id = [i for i in range(1, nb_signers + 1)]
    video_csv_path = os.path.join(tmp_path, "videos.csv")

    for session_nb in range(1, nb_session + 1):
        for record_nb in range(1, nb_records + 1):

            file_prefix = f"CLSFBI{session_nb:02d}{record_nb:02d}"

            eaf_file = f"{file_prefix}.eaf"
            cur_date = datetime.now()
            fps = 50
            frames = random.randint(500, 1500)
            duration = int((frames / fps) * 1000)
            width = 720
            height = 576

            for signer in np.random.choice(signers_id, size=2, replace=False):
                signer_name = f"S{signer:03d}"

                filename = f"{file_prefix}A_{signer_name}_B.mp4"
                filepath = os.path.join(
                    "videos", f"CLSFB - {session_nb:02d} ok", filename
                )

                md5_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()

                annots_left = os.path.join(
                    "annotations", "csv", "left", f"{filename}.csv"
                )

                annots_right = os.path.join(
                    "annotations", "csv", "left", f"{filename}.csv"
                )

                annots_trad = os.path.join(
                    "annotations", "csv", "translation", f"{filename}.csv"
                )

                pose_raw = os.path.join(
                    "features", "landmarks", "pose_raw", f"{filename}.csv"
                )

                hands_raw = os.path.join(
                    "features", "landmarks", "hands_raw", f"{filename}.csv"
                )

                face_raw = os.path.join(
                    "features", "landmarks", "face_raw", f"{filename}.csv"
                )

                pose = os.path.join("features", "landmarks", "pose", f"{filename}.csv")

                hands = os.path.join(
                    "features", "landmarks", "hands", f"{filename}.csv"
                )

                face = os.path.join("features", "landmarks", "face", f"{filename}.csv")

                data["eaf"].append(eaf_file)
                data["session"].append(session_nb)
                data["record"].append(record_nb)
                data["filename"].append(filename)
                data["filepath"].append(filepath)
                data["signer"].append(signer_name)
                data["found"].append(True)
                data["hash"].append(md5_hash)
                data["datetime"].append(cur_date)
                data["fps"].append(fps)
                data["frames"].append(frames)
                data["duration"].append(duration)
                data["width"].append(width)
                data["height"].append(height)
                data["annots_left"].append(annots_left)
                data["annots_right"].append(annots_right)
                data["annots_trad"].append(annots_trad)
                data["pose_raw"].append(pose_raw)
                data["hands_raw"].append(hands_raw)
                data["face_raw"].append(face_raw)
                data["pose"].append(pose)
                data["hands"].append(hands)
                data["face"].append(face)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(video_csv_path)

    return df


def create_landmarks_csv(
    root_dir: str,
    pose_raw_csv_path: str,
    hands_raw_csv_path: str,
    hands_csv_path: str,
    pose_csv_path: str,
    nbr_frame: int,
) -> pd.DataFrame:

    hands_full_path = os.path.join(root_dir, hands_csv_path)
    hands_raw_full_path = os.path.join(root_dir, hands_raw_csv_path)

    pose_full_path = os.path.join(root_dir, pose_csv_path)
    pose_raw_full_path = os.path.join(root_dir, pose_raw_csv_path)

    # Create directories
    os.makedirs(str(PurePath(hands_full_path).parent), exist_ok=True)
    os.makedirs(str(PurePath(hands_raw_full_path).parent), exist_ok=True)
    os.makedirs(str(PurePath(pose_full_path).parent), exist_ok=True)
    os.makedirs(str(PurePath(pose_raw_full_path).parent), exist_ok=True)

    # Create CSV

    hands_data = {}
    pose_data = {}

    for frame in range(nbr_frame):
        for landmark in hands_landmarks_list:

            if landmark not in hands_data:
                hands_data[landmark] = []

            hands_data[landmark].append(random.random())

        for landmark in pose_landmarks_list:

            if landmark not in pose_data:
                pose_data[landmark] = []

            pose_data[landmark].append(random.random())

    hands_df = pd.DataFrame.from_dict(hands_data)
    hands_df.to_csv(hands_full_path)
    hands_df.to_csv(hands_raw_full_path)

    pose_df = pd.DataFrame.from_dict(pose_data)
    pose_df.to_csv(pose_full_path)
    pose_df.to_csv(pose_raw_full_path)

    return hands_df, pose_df


def create_annotations_csv(
    root_dir: str,
    left_hand_csv: str,
    right_hand_csv: str,
    translation_csv: str,
    millisec: int,
):

    left_hand_full_path = os.path.join(root_dir, left_hand_csv)
    right_hand_full_path = os.path.join(root_dir, right_hand_csv)
    translation_full_path = os.path.join(root_dir, translation_csv)

    # Creating folders

    os.makedirs(str(PurePath(left_hand_full_path).parent), exist_ok=True)
    os.makedirs(str(PurePath(right_hand_full_path).parent), exist_ok=True)
    os.makedirs(str(PurePath(translation_full_path).parent), exist_ok=True)

    # Creating annotations
    last_time = 0

    data = {"start": [], "end": [], "gloss": []}

    for time in range(500, millisec, 500):

        data["start"].append(last_time)
        data["end"].append(time)
        data["gloss"].append(f"gloss_{random.randint(1,100)}")

        last_time = time

    gloss_df = pd.DataFrame.from_dict(data)

    gloss_df.to_csv(left_hand_full_path)
    gloss_df.to_csv(right_hand_full_path)
    gloss_df.to_csv(translation_full_path)

    return gloss_df
