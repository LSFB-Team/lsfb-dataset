from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, Callable
import pandas as pd
import os


class LsfbIsolDataset(Dataset):

    """
    Load the LSFB_ISOL dataset.
    The dataset contains video sequence of size 720x576 recorded in 50FPS.

    Those video sequences could be enriched with the following features :
      - Hands landmarks : Media Pipe landamarks for the hands of the signer
      - Face landmarks : Media Pipe landmarks for the face of the signer
      - Pose landmarks : Media Pipe landmarks for the pose of the signer

    Target values :
      - The gloss associated with the video

    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        features: Optional[list[str]] = None,
        labels: Optional[Dict[int, str]] = None,
    ):
        """
        Load the dataset.

        Parameters
        ----------
        root : str
            The root folder containing the dataset
        transform : Optional[Callable]
            The pytorch transform
        features : list[str]
            List of the feature to load
        """

        self.root: str = root
        self.transform = transform
        self.features = features
        self.labels = labels

        self.clips_info = pd.read_csv(os.path.join(root, "clips.csv"))

        if self.features == None:
            self.features = ["video"]

        if self.labels == None:
            self.labels = self._load_label_mapping()

    def _load_label_mapping(self) -> Dict[int, str]:
        """
        Create a mapping between the gloss string and an integer value.
        Returns:

        label_mapping : dictionnary of int : str containing the mapping.

        """
        unique_gloss = self.clips_info["gloss"].unique().tolist()
        label_mapping = dict(map(lambda x: (unique_gloss.index(x), x), unique_gloss))
        return label_mapping
