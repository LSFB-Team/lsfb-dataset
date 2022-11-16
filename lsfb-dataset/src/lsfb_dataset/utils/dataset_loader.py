import pandas as pd
import glob
from tqdm import tqdm
from os.path import sep
from concurrent.futures.process import ProcessPoolExecutor


def load_lsfb_dataset(path: str, verbose: bool = False):
    """
    Read the LSFB corpus in a dataframe containing the label of each signs, a number associated to each label,
    the path of the sign and which subset the signs come from (test or train).

    PARAMETERS:
      path : The path to lsfb corpus folder

    OUTPUT:
      dataset : A dataframe containing 4 columns (label, label_nbr, path, subset)
    """

    signs_folders = glob.glob(f"{path}{sep}*")
    map_label = {value.split(sep)[-1]: idx for (idx, value) in enumerate(signs_folders)}

    dataset = pd.DataFrame(columns=["label", "label_nbr", "path", "subset"])

    results = []
    with ProcessPoolExecutor() as executor:
        for folder in signs_folders:
            results.append(executor.submit(process_sign_folder, folder, map_label))

    df_data = []
    for idx, data in tqdm(enumerate(results), total=len(results), disable=not verbose):
        df_data += data.result()

    return pd.DataFrame(df_data)


def process_sign_folder(folder: str, map_label):
    train_path = f"{folder}{sep}train{sep}*"
    test_path = f"{folder}{sep}test{sep}*"
    label = folder.split(sep)[-1]

    data = []

    for sign in glob.glob(train_path):
        row = {}
        row["label"] = label
        row["label_nbr"] = map_label[label]
        row["path"] = sign
        row["subset"] = "train"

        data.append(row)

    for sign in glob.glob(test_path):
        row = {}
        row["label"] = label
        row["label_nbr"] = map_label[label]
        row["path"] = sign
        row["subset"] = "test"

        data.append(row)

    return data
