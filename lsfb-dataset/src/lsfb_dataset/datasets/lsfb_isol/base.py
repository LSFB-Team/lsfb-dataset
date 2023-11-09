import abc

import pandas as pd

from lsfb_dataset.datasets.lsfb_isol.config import LSFBIsolConfig
from lsfb_dataset.utils.datasets import load_split, load_labels


class LSFBIsolBase:
    def __init__(self, config: LSFBIsolConfig):
        self.config = config

        self.instances: list[str] = load_split(self.config.root, self.config.split)
        self.instance_metadata = pd.read_csv(f"{self.config.root}/instances.csv")

        # sign index path
        sign_index_path = (
            f"{self.config.root}/metadata/{self.config.sign_index_file}.csv"
        )
        self.labels, self.label_to_index, self.index_to_label = load_labels(
            sign_index_path, self.config.n_labels
        )

        self._filter_instances()

        self.targets = {}
        self._load_targets()

    def _filter_instances(self):
        self.instance_metadata = self.instance_metadata[
            self.instance_metadata["id"].isin(self.instances)
        ]
        self.instance_metadata = self.instance_metadata[
            self.instance_metadata["sign"].isin(self.labels)
        ]
        self.instances = self.instance_metadata["id"].tolist()

    def _load_targets(self):
        targets = self.instance_metadata.loc[:, ["id", "sign"]].to_records(index=False)
        if self.config.target == "sign_index":
            self.targets = {key: self.label_to_index[gloss] for key, gloss in targets}
        else:
            self.targets = {key: gloss for key, gloss in targets}

    def __len__(self):
        return len(self.instances)

    @abc.abstractmethod
    def __getitem__(self, index):
        pass
