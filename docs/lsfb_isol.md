# LSFB-ISOL

The LSFB-ISOL dataset pictures isolated signs extracted from the LSFB-CONT dataset. The dataset propose 395 unique gloss with at least 40 occurrences per gloss for a total of 47.551 videos. You can download the dataset on the [official website](https://lsfb.info.unamur.be/).

## Loading the dataset

Once you donwloaded the dataset, you can use this companion library to easily load it into a dataframe with the following code :

```python
from lsfb_dataset.utils.dataset_loader import load_lsfb_dataset


df = load_lsfb_dataset("./dataset/path")

train_split = df[df["subset"] == "train"]
test_split = df[df["subset"] == "test"]
``` 

The loaded dataframe contains 4 columns : 

 - **label** : the gloss of the sign (str)
 - **label_nbr** : Unique number associated with to the gloss (int)
 - **subset** : The dataset is already split into train and test subsets. This column indicates which subset the video belongs to (str)
 - **path** : The path to the video (str)

The dataframe could then be used to create a [Pytorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

## DataLoader (Pytorch)

To dynamically load datasets too big to fit in memory, Pytorch provides a Dataset class that can be used to load the dataset in a streaming fashion. This library provide a custom dataloader for the LSFB-ISOL dataset. 

The `lsfb_dataset.datasets.lsfb_dataset.LsfbIsolDataset` class propose a constructor with the following parameters :

- **df** : The dataframe containing the dataset obtained with the `load_lsfb_dataset` function
- **label_padding** : Required when *sequence_label* is True. The value is a string indicating if the label should be looped or not. The value can be either "loop" or "no_loop". If the labels are not looped, a special padding label is created to fill the sequence.
- **sequence_label** : Boolean value indicating if the labels should be returned as a sequence instead of a single value for the whole video sequence.
- **transforms** : A series of transformations to apply to the video.  
- **max_frame** : The maximum number of frames to extract from the video.
- **labels** : Dictionnary containing a label mapping between each gloss and a unique id. If none the mapping is created from the dataframe.

The following code snippets show how to load the dataset and create a `LsfbIsolDataset` object:

```python
from lsfb_dataset.utils.dataset_loader import load_lsfb_dataset
from lsfb_dataset.datasets.lsfb_dataset import LsfbDataset

df = load_lsfb_dataset("./dataset/path")
train_split = df[df["subset"] == "train"]

train_dataset = LsfbDataset(train_split, max_frame = 100)

```

