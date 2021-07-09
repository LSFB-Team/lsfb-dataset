import pytest
from lsfb_dataset.datasets.lsfb_dataset import LsfbDataset


def test_data_loader_creation(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbDataset(train)

    labels = data_loader.labels

    assert len(labels) == 2


def test_data_loader_creation_sequence_padding(lsfb_isol_dataframe):

    train = lsfb_isol_dataframe[lsfb_isol_dataframe["subset"] == "train"]
    data_loader = LsfbDataset(train, label_padding="zero", sequence_label=True)

    labels = data_loader.labels

    assert len(labels) == 3
