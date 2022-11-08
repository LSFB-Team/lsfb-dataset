# How To Guide

This section provide quick reference for the various class and module available in `lsfb-dataset`. All these examples are available as jupyter noteboks [here](https://github.com/Jefidev/lsfb-dataset/tree/master/examples).

## How to Download Datasets

The LSFB dataset could be downloaded through *HTTP*. The  `lsfb-dataset` package provide a `DatasetDownloader` taking care of the download. If the download is interrupted, the downloader contains mechanism allowing to resume it where it stops.

### Download lsfb Isol landmarks

By default, the downloader will only fetch the landmarks of the lsfb_isol dataset (~5Go). The only mandatory parameter is the destination folder for the dataset. 

```python
from lsfb_dataset.utils.download import DatasetDownloader

downloader = DatasetDownloader("./destination/folder")
downloader.download()
```

### Include Videos

You can easily download the raw video by using the `include video` flag. Be aware that videos are heavy. The size of the lsfb_isol dataset with videos is around 100Go

```python
from lsfb_dataset.utils.download.dataset_downloader import DatasetDownloader

downloader = DatasetDownloader("./destination/folder", include_video=True)
downloader.download()
```

### Downloading lsfb cont dataset

By default, the lsfb_isol dataset is downloaded. To download the continuous dataset, you need to set the mandatory parameter `dataset`.
```python
from lsfb_dataset.utils.download.dataset_downloader import DatasetDownloader

downloader = DatasetDownloader("./destination/folder", dataset="cont", include_video=True)
downloader.download()
```