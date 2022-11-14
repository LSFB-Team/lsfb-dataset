import pytest

import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from fixtures.landmarks import hands_landmarks_list, pose_landmarks_list
import random



@pytest.fixture(scope="session")
def mock_lsfb_isol_path():
    """
    Mock a lsfb isol dataset containing 10 signs label with 30 examples each.
    """
    with tempfile.TemporaryDirectory() as tmp_dirname:

        clips_df = create_dummy_clips_csv(tmp_dirname)
        create_dummy_lemmes_csv(tmp_dirname)


        # Create dummy landmarks files
        for _, row in clips_df.iterrows():

            hands_path = row["hands"]
            pose_path = row["pose"]

            create_landmarks_csv(tmp_dirname, hands_path, pose_path)


        yield tmp_dirname



def create_dummy_clips_csv(root_dir:str, nbr_labels:int=10, nbr_examples:int=30) -> pd.DataFrame:

    csv_path = f"{root_dir}/clips.csv"

    data = {"filename":[], "lemme":[], "class":[], "video":[], "pose":[], "hands": []}

    for label_nbr in range(nbr_labels):
        for example in range(nbr_examples):

            lemme = f"label{label_nbr}"
            lemme_dir = f"_{lemme.upper()}_"
            filename = f"CLSFBI{label_nbr}{example}_S0{example}_B_2_4"

            data["filename"].append(filename +".mp4")
            data["lemme"].append(lemme.upper())
            data["class"].append(label_nbr)

            vid_path = f"videos/{lemme_dir}/{filename}.mp4"
            data["video"].append(vid_path)

            pose_path = f"features/landmarks/{lemme_dir}/pose/{filename}.csv"
            data["pose"].append(pose_path)

            hands_path = f"features/landmarks/{lemme_dir}/hands/{filename}.csv"
            data["hands"].append(hands_path)

    
    df = pd.DataFrame.from_dict(data)
    df.to_csv(csv_path)

    return df


def create_dummy_lemmes_csv(root_dir:str, nbr_labels:int=10, nbr_examples:int=30) -> pd.DataFrame:

    data = {"lemme": [], "count": [], "dir": []}
    lemmes_path = f"{root_dir}/lemmes.csv"

    for label_nbr in range(nbr_labels):
        lemme = f"label{label_nbr}"
        lemme_dir = f"_{lemme.upper()}_"

        data["lemme"].append(lemme)
        data["dir"].append(lemme_dir)
        data["count"].append(nbr_examples)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(lemmes_path)

    return df


def create_landmarks_csv(root_dir:str, hands_csv_path:str, pose_csv_path:str) -> pd.DataFrame:

    hands_full_path = f"{root_dir}/{hands_csv_path}"
    pose_full_path = f"{root_dir}/{pose_csv_path}"


    # Create directories
    output_dir = "/".join(hands_full_path.split("/")[:-1])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir = "/".join(pose_full_path.split("/")[:-1])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Create CSV

    nbr_frame = random.randint(15, 100)

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

    pose_df = pd.DataFrame.from_dict(pose_data)
    pose_df.to_csv(pose_full_path)

    return hands_df, pose_df


    










    












