import pytest
from lsfb_dataset.datasets.lsfb_isol import LSFBIsolConfig, LSFBIsolLandmarks


def test_lsfb_isol_landmarks_loader(mock_lsfb_isol_path_v2):
    """Test the initialization of the LSFBIsolLandmarks class."""
    config = LSFBIsolConfig(mock_lsfb_isol_path_v2)
    isolLoader = LSFBIsolLandmarks(config=config)

    assert len(isolLoader) == 300


def test_lsfb_isol_landmarks_loader_getitem(mock_lsfb_isol_path_v2):
    """Test the __getitem__ method of the LSFBIsolLandmarks class with default params."""
    config = LSFBIsolConfig(mock_lsfb_isol_path_v2)
    isolLoader = LSFBIsolLandmarks(config=config)

    features, targets = isolLoader[0]

    assert len(features) == 3
