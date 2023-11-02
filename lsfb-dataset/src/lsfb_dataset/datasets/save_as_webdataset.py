import pandas as pd
import tempfile
import tarfile
from tqdm import tqdm


def save_as_webdataset(
    instances: pd.DataFrame,
    root: str,
    poses_list: list[str],
    label_to_index: dict[str, int],
    poses_raw: bool,
    output_path: str,
):
    poses_folder = "poses_raw" if poses_raw else "poses"
    data = []

    for _, row in instances.iterrows():
        instance_id = row["id"]

        for pose in poses_list:
            elem_name = f"{instance_id}.{pose}.npy"
            elem_path = f"{root}/{poses_folder}/{pose}/{instance_id}.npy"

            data.append((elem_name, elem_path, "file"))

        sign_class = row["sign"]
        sign_idx = label_to_index[sign_class]

        data.append((f"{instance_id}.class", sign_class, "text"))
        data.append((f"{instance_id}.idx", sign_idx, "text"))

    write_tar_file(data, output_path)


def write_tar_file(data, output_file):
    with tarfile.open(output_file, "w:gz") as tar:
        progress = tqdm(data, desc="Writing tar file")
        for elem in progress:
            if elem[2] == "file":
                elem_name = elem[0]
                elem_path = elem[1]
                tar.add(elem_path, arcname=elem_name)

            elif elem[2] == "text":
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                    f.write(str(elem[1]))

                elem_name = elem[0]
                elem_path = f.name
                tar.add(elem_path, arcname=elem_name)

            else:
                raise ValueError(f"Unknown data type: {elem[2]}")
