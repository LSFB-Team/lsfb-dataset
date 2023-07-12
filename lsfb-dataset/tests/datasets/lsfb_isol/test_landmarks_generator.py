import pytest
from lsfb_dataset.datasets.lsfb_isol import LSFBIsolConfig, LSFBIsolLandmarksGenerator


def test_lsfb_isol_landmarks_loader(mock_lsfb_isol_path_v2):
    """Test the initialization of the LSFBIsolLandmarksGenerator class."""
    config = LSFBIsolConfig(mock_lsfb_isol_path_v2)
    isolLoader = LSFBIsolLandmarksGenerator(config=config)

    assert len(isolLoader) == 300
