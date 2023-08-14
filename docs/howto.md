# How To Guide

This section provide quick reference for the various class and module available in `lsfb-dataset`. All these examples are available as jupyter notebooks [here](https://github.com/Jefidev/lsfb-dataset/tree/master/examples).

## How to Download Datasets

Both LSFB datasets, i.e. `cont` and `isol`, can be downloaded through *HTTP/HTTPS*.
The  `lsfb-dataset` package provide a `Downloader` class taking care of the download according to your needs.
For example, the downloader can filter out the files that you don't need and can also resume the downloading where it stops.

| Name of the dataset | ID   | Poses | Videos (GB) |
|---------------------|------|-------|-------------|
| LSFB ISOL           | isol | 10GB  | 25GB        |
| LSFB CONT           | cont | 31GB  | **~400GB**  |

As you can see in this table, the datasets can be heavy, especially the videos of the LSFB CONT dataset.

### Download LSFB ISOL Landmarks

By default, the downloader will fetch the landmarks of the entirety of the specified dataset. The only mandatory parameters are the dataset name and the destination folder where the files are going to be downloaded.
```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='isol', destination="./destination/folder")
downloader.download()
```

### Download LSFB CONT Landmarks

By default, the downloader will fetch the landmarks of the entirety of the specified dataset. The only mandatory parameters are the dataset name and the destination folder where the files are going to be downloaded.
```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='cont', destination="./destination/folder")
downloader.download()
```

### Include Videos

You can easily download the raw video by using the `include videos` parameter. Be aware that videos are heavy.

```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='isol', destination="./destination/folder", include_videos=True)
downloader.download()
```

```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='cont', destination="./destination/folder", include_videos=True)
downloader.download()
```
