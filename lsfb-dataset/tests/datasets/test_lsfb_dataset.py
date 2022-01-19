import pytest
from lsfb_dataset.datasets.lsfb_isol_dataset import LsfbIsolDataset


def test_data_loader_creation(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbIsolDataset(train)

    labels = data_loader.labels

    assert len(labels) == 2


def test_data_loader_creation_sequence_padding(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbIsolDataset(train, label_padding="zero", sequence_label=True)

    labels = data_loader.labels

    assert len(labels) == 3


def test_video_loading_single_label(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbIsolDataset(train)

    X, y = data_loader[0]

    assert isinstance(y, int)
    assert len(X) == 30


def test_video_loading_sequence_label(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbIsolDataset(train, sequence_label=True)

    X, y = data_loader[0]

    assert len(y) == 30
    assert len(X) == 30

