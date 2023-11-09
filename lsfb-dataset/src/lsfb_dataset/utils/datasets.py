import os
import json
from typing import Optional

import pandas as pd


def load_split(root: str, split_name: str) -> list[str]:
    with open(
        os.path.join(root, "metadata", "splits", f"{split_name}.json"), "r"
    ) as file:
        return json.load(file)


def load_labels(root: str, n_labels: Optional[int] = None):
    signs = pd.read_csv(f"{root}/metadata/sign_to_index.csv").to_records(index=False)
    labels = []
    label_to_index = {}
    index_to_label = {}

    for sign, sign_index in signs:
        print(n_labels, type(n_labels))
        if n_labels is not None and sign_index >= n_labels:
            sign_index = -1
            index_to_label[-1] = "OTHER_SIGN"
        else:
            labels.append(sign)
            index_to_label[sign_index] = sign
        label_to_index[sign] = sign_index

    return labels, label_to_index, index_to_label
