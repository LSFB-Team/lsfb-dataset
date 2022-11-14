import pytest
from lsfb_dataset.datasets.lsfb_isol import LSFBIsolConfig, LSFBIsolLandmarks


def test_lsfb_isol_loader(mock_lsfb_isol_path):

    config = LSFBIsolConfig(mock_lsfb_isol_path)
    isolLoader = LSFBIsolLandmarks(config = config)

    assert len(isolLoader) == 300