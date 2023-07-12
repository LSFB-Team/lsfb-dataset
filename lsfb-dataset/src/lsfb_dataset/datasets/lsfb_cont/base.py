import abc
import json
import logging
from os import path
from math import floor, ceil

import pandas as pd

from lsfb_dataset.datasets.lsfb_cont.config import LSFBContConfig
from lsfb_dataset.utils.datasets import load_split, load_labels


class LSFBContBase:

    def __init__(self, config: LSFBContConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.logger.info('-' * 10 + "LSFB CONT DATASET")

        self.instances: list[str] = load_split(self.config.root, self.config.split)
        self.instance_metadata = pd.read_csv(path.join(config.root, 'instances.csv'))
        self.instance_metadata = self.instance_metadata[self.instance_metadata['id'].isin(self.instances)]

        self.labels, self.label_to_index, self.index_to_label = load_labels(self.config.root, self.config.n_labels)

        self.annotations: dict[str] = {}
        self._load_annotations()

        self.windows = None
        if self.config.window:
            self._make_windows()

    def _init_logger(self):
        handler = logging.StreamHandler()
        if self.config.verbose:
            handler.setLevel(logging.CRITICAL)
        else:
            handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _transform_sign_annotation(self, annotation: dict[str]):
        start, end, label = int(annotation['start']), int(annotation['end']), annotation['gloss']
        if self.config.segment_unit == 'frame':
            start, end = floor(start/20), ceil(end/20)
        if self.config.segment_label == 'sign_index':
            label = self.label_to_index[label]
        elif self.config.segment_label == 'text':
            label = label.lower()
        return start, end, label

    def _load_annotations(self):
        prefix = self.config.segment_level
        suffix = "merged_hands" if self.config.hands == 'both' else self.config.hands
        with open(f"{self.config.root}/targets/{prefix}_{suffix}.json", 'r') as file:
            all_annotations = json.load(file)
        for instance_id in self.instances:
            annotations = all_annotations[instance_id]
            if self.config.segment_level == 'signs':
                self.annotations[instance_id] = pd.DataFrame.from_records(
                    [self._transform_sign_annotation(a) for a in annotations],
                    columns=['start', 'end', 'label'],
                )
            elif self.config.segment_level == 'subtitles':
                raise NotImplementedError("Subtitles are not yet implemented.")  # TODO

    # def _load_labels(self):
    #     root = self.config.root
    #     n_labels = self.config.n_labels
    #     signs = pd.read_csv(f'{root}/metadata/sign_to_index.csv').to_records(index=False)
    #     for sign, sign_index in signs:
    #         if sign_index >= n_labels:
    #             sign_index = -1
    #             self.index_to_label[-1] = 'OTHER_SIGN'
    #         else:
    #             self.labels.append(sign)
    #             self.index_to_label[sign_index] = sign
    #         self.label_to_index[sign] = sign_index

    def _make_windows(self):
        window_size, window_stride = self.config.window
        self.windows = []
        for instance_id, n_frames in self.instance_metadata[['id', 'n_frames']].to_records(index=False):
            for start in range(0, n_frames, window_stride):
                end = min(start + window_size, n_frames - 1)
                self.windows.append((instance_id, start, end))

    def _apply_transforms(self, features, annotations):
        if self.config.features_transform:
            features = self.config.features_transform(features)
        if self.config.target_transform:
            annotations = self.config.target_transform(annotations)
        if self.config.transform:
            features, annotations = self.config.transform(features, annotations)
        return features, annotations

    def __len__(self):
        if self.config.window is None:
            return len(self.instances)
        return len(self.windows)

    def __getitem__(self, index):
        if self.windows is None:
            return self.__get_instance__(index)
        return self.__get_window__(index)

    @abc.abstractmethod
    def __get_instance__(self, index):
        pass

    @abc.abstractmethod
    def __get_window__(self, index):
        pass
