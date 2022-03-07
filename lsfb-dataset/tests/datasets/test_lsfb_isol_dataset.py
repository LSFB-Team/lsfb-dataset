import pytest
from lsfb_dataset.datasets.lsfb_isol.lsfb_isol_dataset import LsfbIsolDataset


def test_constructor(lsfb_isol_path):
    ds = LsfbIsolDataset(lsfb_isol_path)

    assert len(ds) == 9
