import pytest
from lsfb_dataset.utils.datasets import load_split, load_labels


def test_load_split(mock_lsfb_isol_path_v2):
    id_list = load_split(mock_lsfb_isol_path_v2, "all")
    assert len(id_list) == 300


def test_load_labels(mock_lsfb_isol_path_v2):
    label, lab_to_idx, idx_to_lab = load_labels(mock_lsfb_isol_path_v2)

    assert len(label) == 10
    assert len(lab_to_idx) == 10
    assert len(idx_to_lab) == 10

    assert lab_to_idx[label[0]] == 0
    assert idx_to_lab[0] == label[0]


def test_load_labels_optional_param(mock_lsfb_isol_path_v2):
    label, lab_to_idx, idx_to_lab = load_labels(mock_lsfb_isol_path_v2, 5)

    assert len(label) == 5
    assert len(lab_to_idx) == 10
    assert len(idx_to_lab) == 6

    assert lab_to_idx[label[0]] == 0
    assert idx_to_lab[0] == label[0]
    assert idx_to_lab[-1] == "OTHER_SIGN"
