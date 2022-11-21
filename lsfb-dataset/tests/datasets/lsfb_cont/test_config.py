import pytest

from lsfb_dataset.datasets.lsfb_cont import LSFBContConfig, LSFBContLandmarks


def test_lsfb_cont_landmarks_config(mock_lsfb_cont_path):

    config = LSFBContConfig(mock_lsfb_cont_path, split="train")
    loader = LSFBContLandmarks(config=config)
