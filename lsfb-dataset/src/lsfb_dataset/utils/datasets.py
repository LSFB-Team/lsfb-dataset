import os
import json

import pandas as pd
import numpy as np


def load_split(root: str, split_name: str) -> list[str]:
    with open(os.path.join(root, 'metadata', 'splits', f'{split_name}.json'), 'r') as file:
        return json.load(file)


def split_isol(dataframe: pd.DataFrame, test_frac=0.25, seed=42):
    test_df = dataframe.sample(frac=test_frac, random_state=seed)
    train_df = dataframe.drop(index=test_df.index)
    return train_df, test_df


def create_mask(seq_len: int, padding: int, mask_value: int = 0):
    mask = np.ones(seq_len, dtype='uint8')
    mask[-padding:] = mask_value
    return mask
