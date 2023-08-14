# Download Datasets

Both LSFB datasets, i.e. `cont` and `isol`, can be downloaded through *HTTP/HTTPS*.
The  `lsfb-dataset` package provide a `Downloader` class taking care of the download according to your needs.
For example, the downloader can filter out the files that you don't need and can also resume the downloading where it stops.

| Name of the dataset | ID   | Poses | Videos (GB) |
|---------------------|------|-------|-------------|
| LSFB ISOL           | isol | 10GB  | 25GB        |
| LSFB CONT           | cont | 31GB  | **~400GB**  |

As you can see in this table, the datasets can be heavy, especially the videos of the LSFB CONT dataset.

## Basic Usage

### Download LSFB ISOL Landmarks and Videos

By default, the downloader will fetch the landmarks of the entirety of the specified dataset. The only mandatory parameters are the dataset name and the destination folder where the files are going to be downloaded.
```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='isol', destination="./destination/folder", include_videos=True)
downloader.download()
```

### Download LSFB CONT Landmarks and Videos

By default, the downloader will fetch the landmarks of the entirety of the specified dataset. The only mandatory parameters are the dataset name and the destination folder where the files are going to be downloaded.
```python
from lsfb_dataset import Downloader

downloader = Downloader(dataset='cont', destination="./destination/folder", include_videos=True)
downloader.download()
```

## A more complex example

Here's a more complex example where we only download the instances:
* Of the subsets `train`, `fold_0` and `fold_2`;
* Only the instances of the signers `20 to 39`;
* Only download the raw poses (without any interpolation of the missing landmarks nor smoothing);
* Only includes the landmarks of the `pose` (body) and the hands;
* Without skipping the existing files.
```python
from lsfb_dataset import Downloader

downloader = Downloader(
    dataset='isol',
    destination="./destination/folder",
    splits=['train', 'fold_0', 'fold_2'],
    signers=list(range(20, 40)),
    include_cleaned_poses=False,
    include_raw_poses=True,
    include_videos=False,
    landmarks=['pose', 'left_hand', 'right_hand'],
    skip_existing_files=False,
)
downloader.download()
```
