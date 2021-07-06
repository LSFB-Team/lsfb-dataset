# LSFB Dataset

This library is a companion for the [French Belgian Sign Language dataset](https://lsfb.info.unamur.be/). You will find useful functions to load and manipulate the video from the LSFB dataset. The package provide a pytorch dataset class and several useful transformations methods for video.

*TODO impelement the deploy here https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure*

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

train_dataset = LsfbDataset(train_split, max_frame = 100)

```

For each video, the dataset loader will extract up 100 frames for each video. and return the loaded video along with its numerical label. Other parameters could be added to change the output of the data loader such as :

- **one_hot** : boolean value indicating if the label should be returned [one hot encoded](https://wikipedia.org/wiki/Encodage_one-hot)
- **sequence_label** : boolean value indicating if the labels should be returned as a sequence instead of a signle value (one label for each frame)
- **transforms** : A series of transformations to apply to the video

## Transformations

Transformations are inspired by [torchvision](https://pytorch.org/vision/stable/transforms.html). You can compose a series of tranformation to apply to the video. The dataset loader transform all the video before returning them.

```python
from torchvision import transforms
from lsfb_dataset.transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
    I3DPixelsValue,
    RandomTrimVideo,
    TrimVideo,
    PadVideo,
)
from lsfb_dataset.datasets.lsfb_dataset import LsfbDataset


composed_train = transforms.Compose(
    [
        RandomTrimVideo(nbr_frames),
        PadVideo(nbr_frames),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        I3DPixelsValue(),
        ChangeVideoShape("CTHW"),
    ]
)

train_dataset = LsfbDataset(train_split, transforms=composed_train, max_frame = 100)

```

The available transformations are the following and are built upon [this project](https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/functional.py).

### I3DPixelsValue

By default, the dataloader normalize the pixel value between 0 and 1. This transformation will change that to normalize the pixels value between -1 and 1 like in the I3D paper.

### ChangeVideoShape

Expect to receive tha video in the shape (Time, Height, Width, Channels) which is the default format of cv2 or PIL and change this shape to either channel first CTHW (Channels, Time, Height, Width) or time first TCHW format.

### ResizeVideo

Resize a video in shape (T, H, W, C) to the desired size.

### RandomCropVideo

Crop randomly a video in shape (T, H, W, C) to the desired size.

### CenterCropVideo

Crop a video in shape (T, H, W, C) at its center.

### TrimVideo

Reduce the length of the video to the desired number of frames. 

### RandomTrimVideo

Trim randomly the video to the desired length. This could be use as a type of data augmentation

### PadVideo 

Pad the video to the desired length. The padding could eather be made by looping the video or by adding zero frame at the end.
