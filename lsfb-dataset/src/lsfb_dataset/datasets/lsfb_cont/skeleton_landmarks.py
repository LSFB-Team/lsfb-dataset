import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from ...utils.annotations import annotations_to_vec, create_coerc_vec


class SkeletonLandmarksDataset(Dataset):

    def __init__(self, root_dir: str, video_information: pd.DataFrame, transform=None, isolate_transitions=False):
        self.root = root_dir
        self.video_information = video_information[[
            'frames_nb',
            'right_hand_annotations',
            'left_hand_annotations',
            'upper_skeleton'
        ]].dropna()
        self.transform = transform
        self.transitions = isolate_transitions

    def __len__(self):
        return self.video_information.shape[0]

    def __getitem__(self, index):
        video = self.video_information.iloc[index]
        features = pd.read_csv(os.path.join(self.root, video['upper_skeleton'])).values

        annot_right = pd.read_csv(os.path.join(self.root, video['right_hand_annotations']))
        annot_left = pd.read_csv(os.path.join(self.root, video['left_hand_annotations']))
        classes = annotations_to_vec(annot_right, annot_left, int(video['frames_nb']))

        if self.transitions:
            classes = create_coerc_vec(classes)

        if self.transform:
            features = self.transform(features)

        return features, classes
