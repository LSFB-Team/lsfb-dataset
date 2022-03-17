from torch.utils.data import Dataset

from PIL import Image
import os
from tqdm.notebook import tqdm

from typing import Optional, Callable


def _check_feature(feature: str):
    """
    Check the name of a feature of the LSFB CONT image folder dataset.
    The available features are :
    - right_hand
    - left_hand
    - face

    Parameters
    ----------
    feature : str
        The feature name that is checked

    Raises
    ------
    ValueError if the feature is unknown.
    """

    if feature not in ["right_hand", "left_hand", "face"]:
        raise ValueError(f"Unknown feature: {feature}")


def _get_sample_size(sample: dict) -> int:
    """
    Compute the size of a sample of the dataset.

    Parameters
    ----------
    sample : dict
        The sample of the dataset

    Returns
    -------
    int
        The size of the sample

    Raises
    ------
    ValueError if the sample has no feature
    ValueError if there are different numbers of items in the features of the sample
    """

    keys = list(sample.keys())
    if len(keys) < 1:
        raise ValueError("Sample has no feature.")

    size = len(sample[keys[0]])
    for k in keys:
        if len(sample[k]) != size:
            raise ValueError(
                f"Different number of items in features ({size} in {keys[0]} and {len(sample[k])} in {k})."
            )

    return size


def _get_category_name(cat_nb: int) -> str:
    """
    Return the name of a video category from its number.
    Example : 4 -> 'CLSFB - 04 ok'

    Parameters
    ----------
    cat_nb : int
        The number of the category

    Returns
    -------
    str
        The name of the category
    """
    return f"CLSFB - {cat_nb:02d} ok"


def _get_class_index(filename: str) -> int:
    """
    Return the index of the class of the instance.

    Parameters
    ----------
    filename : str
        The filename used to fetch the class

    Returns
    -------
    int
        The index of the class
    """
    return 1 if filename[-5] == "W" else 0


def _make_dataset(directory: str, categories: list[str], features: list[str]) -> dict:
    """
    Make the dataset. A sample containing images locations per feature is built.
    Only images from the specified categories are fetched.
    If a category is missing in the dataset folder, it is prompted and skipped.

    {feature_1: [image paths], feature_2: [image paths], ... }

    It is recommended to check the sample size later.

    Parameters
    ----------
    directory : str
        The folder containing all the LSFB_CONT image files
    categories : list[str]
        The list of the categories used to fetch the images for the sample
    features : list[str]
        The list of fetched features

    Returns
    -------
    dict
        The built sample {feature_1: [image paths], feature_2: [image paths], ... }

    Raises
    ------
    ValueError if the folder containing the images of a feature is not found
    """
    print("Loading dataset...")

    pbar = tqdm(categories, unit="cat")

    instances = {x: [] for x in features}

    for cat in pbar:
        cat_dir = os.path.join(directory, cat)
        if not os.path.isdir(cat_dir):
            pbar.write(f'Category "{cat}" not found. Skipped.')
        else:
            # noinspection PyUnresolvedReferences
            for video in sorted(
                entry.name for entry in os.scandir(cat_dir) if entry.is_dir()
            ):
                pbar.set_description(f"[ {cat} / {video} ]")
                for feat in features:
                    video_dir = os.path.join(cat_dir, video, feat)

                    if not os.path.isdir(video_dir):
                        raise ValueError(f"Feature folder not found: {video_dir}")

                    for root, _, fnames in sorted(os.walk(video_dir, followlinks=True)):
                        for fname in fnames:
                            path = os.path.join(root, fname)
                            item = path, _get_class_index(fname)
                            instances[feat].append(item)
    pbar.close()
    return instances


def _load_image(img_path: str) -> Image:
    """
    Load an image file into a PIL image object

    Parameters
    ----------
    img_path : str
        The filepath of the image

    Returns
    -------
    Image.pyi
        The image object
    """
    with open(img_path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageDataset(Dataset):
    """
    Load the LSFB_CONT Dataset from an image folder.
    This folder must contain 224x224 images for each frame of all the sequences.

    Example of the path of an image : root/CLSFB - 01 ok/CLSFBI0103A_S001_B.mp4/right_hand/1245W.jpg

    It is an image of the feature 'right_hand' of the 1245th frame
    of the 'CLSFBI0103A_S001_B' video in the 'CLSFB - 01 ok' category.
    The W in the filename means that the label of this instance is 'talking'.

    Supported features:
    - right_hand : The right hand of the speaker
    - left_hand : The left hand of the speaker
    - face : The face of the speaker

    Target values:
    - waiting : The speaker is waiting
    - talking : The speaker is talking

    Properties
    ----------
    root : str
        The root folder containing the dataset
    transform : Optional[Callable]
        The pytorch transform
    categories : list[str]
        The categories used to make the sample
    features : list[str]
        The features used in the sample
    class_names : list[str]
        The names of the classes in the dataset

    sample : dict
        The sample containing the image paths per feature
    size : int
        The size of the sample
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        categories_nb: Optional[int] = 50,
        categories_offset: Optional[int] = 0,
        features: Optional[list[str]] = None,
    ):
        """
        Load the dataset.

        Parameters
        ----------
        root : str
            The root folder containing the dataset
        transform : Optional[Callable]
            The pytorch transform
        categories_nb : int
            The number of categories fetched
        categories_offset : int
            The offset used for fetching categories
        features : list[str]
            The features used in the dataset. If None, set to ['right_hand']
        """
        self.root: str = root
        self.transform = transform
        self.categories = [
            _get_category_name(n)
            for n in range(categories_offset + 1, categories_offset + categories_nb + 1)
        ]

        if features is None:
            features = ["right_hand"]

        self.features = features
        self.class_names = ["waiting", "talking"]
        self.sample = _make_dataset(root, self.categories, features)
        self.size = _get_sample_size(self.sample)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = {}
        for f in self.features:
            if self.transform is not None:
                item[f] = self.transform(_load_image(self.sample[f][index][0]))
            else:
                item[f] = _load_image(self.sample[f][index][0])
            item[f"{f}_label"] = self.sample[f][index][1]
        return item
