import pandas as pd
import numpy as np


def split_cont(df_videos: pd.DataFrame, signers_frac=0.6, seed=42):
    signers = pd.Series(df_videos['signer'].unique())
    train_signers = signers.sample(frac=signers_frac, random_state=seed)
    val_signers = signers.drop(index=train_signers.index)
    train_df = df_videos[df_videos['signer'].isin(train_signers)]
    val_df = df_videos[df_videos['signer'].isin(val_signers)]
    return train_df, val_df


def split_isol(dataframe: pd.DataFrame, test_frac=0.25, seed=42):
    test_df = dataframe.sample(frac=test_frac, random_state=seed)
    train_df = dataframe.drop(index=test_df.index)
    return train_df, test_df


def mini_sample(dataframe: pd.DataFrame, num_samples: int = 10, seed=42):
    return dataframe.sample(n=num_samples, random_state=seed)


def create_mask(seq_len: int, padding: int, mask_value: int = 0):
    mask = np.ones(seq_len, dtype='uint8')
    mask[-padding:] = mask_value
    return mask
