import pytest

import os
import tempfile
import numpy as np
import pandas as pd
import json


@pytest.fixture(scope="session")
def mock_lsfb_isol_path_v2() -> str:
    """
    Mock a lsfb isol dataset containing 10 signs label with 30 examples each.
    """

    print("yo")

    with tempfile.TemporaryDirectory() as tmp_dirname:
        df = create_dummy_instances_csv(tmp_dirname)
        create_dummy_metadata(tmp_dirname, df)
        yield tmp_dirname


def create_dummy_instances_csv(
    root_dir: str, nbr_labels: int = 10, nbr_examples: int = 30
) -> pd.DataFrame:
    csv_path = f"{root_dir}/instances.csv"

    data = {
        "id": [],
        "sign": [],
        "signer": [],
        "start": [],
        "end": [],
    }

    for label_nbr in range(nbr_labels):
        for example in range(nbr_examples):
            start = 2500
            end = 2600

            sign = f"label{label_nbr}"
            clip_id = (
                f"CLSFBI{label_nbr:02d}{example:02d}_S0{example:02d}_B_{start}_{end}"
            )
            signer = f"S0{example}"

            data["id"].append(clip_id)
            data["sign"].append(sign)
            data["signer"].append(signer)
            data["start"].append(start)
            data["end"].append(end)

            for pose_file in ["poses", "poses_raw"]:
                create_pose_files(root_dir, clip_id, pose_file)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(csv_path)

    return df


def create_pose_files(
    root_dir: str, clip_id: str, pose_folder: str, nb_frame: int = 25
) -> None:
    pose_path = f"{root_dir}/{pose_folder}"

    if os.path.exists(pose_path) == False:
        os.mkdir(pose_path)

    sub_folders = [
        {"name": "pose", "shape": (nb_frame, 33, 3)},
        {"name": "right_hand", "shape": (nb_frame, 21, 3)},
        {"name": "left_hand", "shape": (nb_frame, 21, 3)},
        {"name": "face", "shape": (nb_frame, 478, 3)},
    ]

    for sub_folder in sub_folders:
        sub_folder_path = f"{pose_path}/{sub_folder['name']}"

        if os.path.exists(sub_folder_path) == False:
            os.mkdir(sub_folder_path)

        mocked_landmarks = np.random.rand(*sub_folder["shape"])
        landmark_file = f"{sub_folder_path}/{clip_id}.npy"

        np.save(landmark_file, mocked_landmarks)


def create_dummy_metadata(root_dir: str, sign_df: pd.DataFrame) -> None:
    metadata_dir = f"{root_dir}/metadata"

    if not os.path.exists(metadata_dir):
        os.mkdir(metadata_dir)

    # Sign occurence csv
    sign_occurence = sign_df.groupby("sign").count()["id"]
    sign_occurence.to_csv(
        f"{metadata_dir}/sign_occurences.csv",
        header=["count"],
    )

    # sign to index csv
    unique_signs = sign_df["sign"].unique()
    idx_tuples = [(sign, idx) for idx, sign in enumerate(unique_signs)]
    index_df = pd.DataFrame(idx_tuples, columns=["sign", "class"])
    index_df.to_csv(f"{metadata_dir}/sign_to_index.csv", index=False)

    # Get
    sign_id = sign_df["id"].unique()
    sign_amount = len(sign_id)

    # splits directory
    splits_dir = f"{metadata_dir}/splits"
    if not os.path.exists(splits_dir):
        os.mkdir(splits_dir)

    # All json
    with open(f"{splits_dir}/all.json", "w") as f:
        json.dump(sign_id.tolist(), f)

    # Train json
    with open(f"{splits_dir}/train.json", "w") as f:
        json.dump(sign_id[: int(sign_amount * 0.8)].tolist(), f)

    # test json
    with open(f"{splits_dir}/test.json", "w") as f:
        json.dump(sign_id[int(sign_amount * 0.8) :].tolist(), f)

    # mini sample json
    with open(f"{splits_dir}/mini_sample.json", "w") as f:
        json.dump(sign_id[:10].tolist(), f)

    # splits  5 folds
    for fold in range(5):
        with open(f"{splits_dir}/fold_{fold}.json", "w") as f:
            json.dump(sign_id[fold::5].tolist(), f)
