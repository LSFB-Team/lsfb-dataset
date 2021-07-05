# LSFB Dataset

This library is a companion for the [French Belgian Sign Language dataset](https://lsfb.info.unamur.be/). You will find useful functions to load and manipulate the video from the LSFB dataset. The package provide a pytorch dataset class and several useful transformations methods for video.

## Loading the dataset

Currently, only the loading of LSFB-isol is supported. The dataset needs to be downloaded from [the website](https://lsfb.info.unamur.be/). The dataset information could than be read into a dataframe like this :

```python
from lsfb_dataset.utils.dataset_loader import load_lsfb_dataset


df = load_lsfb_dataset("./dataset/path")

train_split = df[df["subset"] == "train"]
test_split = df[df["subset"] == "test"]
```

## Dataset Loader

If you are using pytorch, you can use the Dataloader provided by this library in order to feed the video into your model. If you are not familiar with the concept of pytorch dataset and dataloaders please check out the [pytorch documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### Basic Usage

```python
from lsfb_dataset.utils.dataset_loader import load_lsfb_dataset
from lsfb_dataset.datasets.lsfb_dataset import LsfbDataset

df = load_lsfb_dataset("./dataset/path")
train_split = df[df["subset"] == "train"]

train_dataset = LsfbDataset(train_split)

```



