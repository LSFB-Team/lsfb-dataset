import pytest

import os
import tempfile
import numpy as np
import pandas as pd
import json


@pytest.fixture(scope="session")
def mock_lsfb_cont_path_v2() -> str:
    """
    Mock the LSFB cont dataset
    """
    with tempfile.TemporaryDirectory() as tmp_dirname:
        create_dummy_targets(tmp_dirname)

        yield tmp_dirname


def create_dummy_targets(tmp_dirname: str, nbr_videos: int = 10, nbr_signs: int = 30):
    targets_folder = f"{tmp_dirname}/targets"

    if not os.path.exists(targets_folder):
        os.mkdir(targets_folder)

    files_name = [
        "signs_left_hand.json",
        "signs_merged_hands.json",
        "signs_right_hands.json",
        "special_signs_left_hand.json",
        "special_signs_merged_hands.json",
        "special_signs_right_hands.json",
    ]

    for target_file in files_name:
        data_dict = {}

        for vid in nbr_videos:
            vid_id = f"CLSFBI01{vid:02d}A_S0{vid:02d}_B"
            data_dict[vid_id] = []
            time = 1000

            for sign in nbr_signs:
                sign = f"label{sign}"

                sign_dict = {"start": time, "end": time + 100, "label": sign}
                data_dict[vid_id].append(sign_dict)

                time += 100

            for pose_folder in ["poses", "poses_raw"]:
                create_dummy_pose_files(tmp_dirname, pose_folder, vid_id)

        # Dump json
        target_path = f"{targets_folder}/{target_file}"
        with open(target_path, "w") as f:
            json.dump(data_dict, f)


def create_dummy_pose_files(
    tmp_dirname: str, pose_folder: str, vid_id: str, len_vid: int = 5000
):
    pose_path = f"{tmp_dirname}/{pose_folder}"

    if not os.path.exists(pose_path):
        os.mkdir(pose_path)

    sub_folders = [
        {"name": "pose", "shape": (len_vid, 33, 3)},
        {"name": "right_hand", "shape": (len_vid, 21, 3)},
        {"name": "left_hand", "shape": (len_vid, 21, 3)},
        {"name": "face", "shape": (len_vid, 478, 3)},
    ]

    for sub_folder in sub_folders:
        sub_folder_path = f"{pose_path}/{sub_folder['name']}"

        if os.path.exists(sub_folder_path) == False:
            os.mkdir(sub_folder_path)

            mocked_landmarks = np.random.rand(*sub_folder["shape"])
            landmark_file = f"{sub_folder_path}/{vid_id}.npy"

            np.save(landmark_file, mocked_landmarks)
