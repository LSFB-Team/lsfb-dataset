# LSFB-ISOL

The LSFB-ISOL dataset pictures isolated signs extracted from the LSFB-CONT dataset. The dataset propose 635 unique gloss with at least 40 occurrences per gloss for a total of 54.551 videos. You can download the dataset on the [official website](https://lsfb.info.unamur.be/).

## Loading the dataset

Once you've donwloaded the dataset, you can use this companion library to easily load it into a dataframe with the following code :

```python
from lsfb_dataset.datasets.lsfb_isol.lsfb_isol_dataset import LsfbIsolDataset

dataset_dir = "./lsfb_isol"

dataset = LsfbIsolDataset(dataset_dir)

``` 

This class creates a [pytorch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). The constructor of the class takes the following arguments: 

 - **root** : The path to the root directory of the dataset. `str`
 - **transforms** : A list of transformation to apply on the input data (see [pytorch transform](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)). `Optional[List[Callable]]`
 - **features** : List of the features you want to load. By default only the video frame are loaded. The possible value are : Hand landmarks, Face landmarks, Pose landmarks and video. The landmarks are provided by [mediapipe](https://mediapipe.dev/) `Optional[List[str]]`
 - **labels** : Allows you to provide a custom label mapping between the string label of the gloss and a numerical label. If not provided the mapping is automatically created. `Optional[Dict[str, int]]`
 - **max_frames** : The maximum number of frames to load for each video. (default=50)`Optional[int]`

The dataset could be iterated and return a tuple containing the numeric label of the considered gloss and a dictionnary containing a key for each loaded feature and a numpy array containing the data for this feature.

## DataLoader (Pytorch)

The dataset could then be used with a [pytorch dataloader](https://pytorch.org/docs/stable/data.html) to load the data in a batch for the training.
