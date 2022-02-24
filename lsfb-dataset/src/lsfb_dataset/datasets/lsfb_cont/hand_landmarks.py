from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

from ...utils.annotations import annotations_to_vec


class HandPositionDataset(Dataset):
    def __init__(self, root_dir, video_information: pd.DataFrame, transform=None, num_videos=None):
        self.root_dir = root_dir
        self.video_information = video_information[
            ['frames_nb', 'right_hand_annotations', 'holistic_landmarks_clean']].dropna()

        if num_videos is not None:
            self.video_information = self.video_information.sample(n=num_videos)

        self.transform = transform

        self.class_names = ['waiting', 'talking']
        self.class_proportion = [0.69, 0.31]

    def __len__(self):
        return self.video_information.shape[0]

    def __getitem__(self, index):
        vid_info = self.video_information.iloc[index]
        df_holistic = pd.read_csv(os.path.join(self.root_dir, vid_info['holistic_landmarks_clean']))
        df_annot = pd.read_csv(os.path.join(self.root_dir, vid_info['right_hand_annotations']))

        features = df_holistic[['RIGHT_HAND_X', 'RIGHT_HAND_Y', 'LEFT_HAND_X', 'LEFT_HAND_X']].values
        classes = annotations_to_vec(df_annot, None, int(vid_info['frames_nb']))

        if self.transform is not None:
            features = self.transform(features).float()

        return features, classes


class HandMovementDataset(Dataset):
    def __init__(self, root_dir, video_information: pd.DataFrame, transform=None, num_videos=None):
        self.root_dir = root_dir
        self.video_information = video_information[
            ['frames_nb', 'right_hand_annotations', 'holistic_landmarks_clean']].dropna()

        if num_videos is not None:
            self.video_information = self.video_information.sample(n=num_videos)

        self.transform = transform

        self.class_names = ['waiting', 'talking']
        self.class_proportion = [0.69, 0.31]

    def __len__(self):
        return self.video_information.shape[0]

    def __getitem__(self, index):
        vid_info = self.video_information.iloc[index]
        df_holistic = pd.read_csv(os.path.join(self.root_dir, vid_info['holistic_landmarks_clean']))
        df_annot = pd.read_csv(os.path.join(self.root_dir, vid_info['right_hand_annotations']))

        hand_pos = df_holistic[['RIGHT_HAND_X', 'RIGHT_HAND_Y', 'LEFT_HAND_X', 'LEFT_HAND_X']]
        hand_vel = hand_pos.diff(periods=10).fillna(method='bfill')
        hand_acc = hand_vel.diff(periods=10).fillna(method='bfill')

        features = np.concatenate((hand_pos.values, hand_vel.values, hand_acc.values), axis=1)
        classes = annotations_to_vec(df_annot, None, int(vid_info['frames_nb']))

        if self.transform is not None:
            features = self.transform(features)

        return features, classes
