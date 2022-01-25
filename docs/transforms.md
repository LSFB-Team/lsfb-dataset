# Transforms

The transforms class are inspired from the [TorchVision transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) workflow. Unfortunately, torchvision does not (yet) provides transforms for video. To help you to easily build your own pre-processing or data augmentation workflow we created several extension of torchvision transforms for video.

## How to Use Transforms

The transfoms must be imported and could be chain as you like :

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

## Available Transforms

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

