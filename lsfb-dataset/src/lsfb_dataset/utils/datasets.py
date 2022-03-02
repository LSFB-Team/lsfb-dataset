import pandas as pd
import os
from tqdm.auto import tqdm
from .annotations import annotations_to_vec, create_coerc_vec


def make_windows(videos: pd.DataFrame, window_size: int, stride: int):
    frames = []
    # (video_idx, start, off)

    for idx, video in videos.iterrows():
        frames_nb = int(video['frames_nb'])
        for f in range(0, frames_nb, stride):
            frames.append((idx, f, f + window_size))

    return frames


def load_data(root: str, feature_name, videos: pd.DataFrame, isolate_transition=False):
    print(f'Loading {feature_name} and classes...')
    data = {}

    for idx, video in tqdm(videos.iterrows(), total=videos.shape[0]):
        features = pd.read_csv(os.path.join(root, video[feature_name])).values

        annot_right = pd.read_csv(os.path.join(root, video['right_hand_annotations']))
        annot_left = pd.read_csv(os.path.join(root, video['left_hand_annotations']))
        classes = annotations_to_vec(annot_right, annot_left, int(video['frames_nb']))
        if isolate_transition:
            classes = create_coerc_vec(classes)

        data[idx] = features, classes
    return data
