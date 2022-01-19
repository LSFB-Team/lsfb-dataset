from lsfb_dataset.utils.dataset_loader import load_lsfb_dataset


def test_loading_dataset(lsfb_isol_path):
    df = load_lsfb_dataset(lsfb_isol_path)

    assert len(df) == 16

    test = df[df["subset"] == "test"]
    train = df[df["subset"] == "train"]

    assert len(test) == 8
    assert len(train) == 8

