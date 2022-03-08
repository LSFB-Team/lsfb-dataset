import pytest
from lsfb_dataset.datasets.lsfb_isol.lsfb_isol_dataset import LsfbIsolDataset


def test_constructor(lsfb_isol_path):
    ds = LsfbIsolDataset(lsfb_isol_path)

    assert len(ds) == 9
    assert len(ds.labels) == 3


def test_label_mapping_setter(lsfb_isol_path):

    mapping = {"GLOSS1": 1, "GLOSS2": 2, "GLOSS3": 3, "GLOSS4": 4}
    ds = LsfbIsolDataset(lsfb_isol_path, labels=mapping)

    assert len(ds.labels) == 4


def test_data_iteration(lsfb_isol_path):

    ds = LsfbIsolDataset(lsfb_isol_path)

    for X, y in ds:
        assert y in {v: k for k, v in ds.labels.items()}
        assert X["video"].shape == (30, 224, 224, 3)
