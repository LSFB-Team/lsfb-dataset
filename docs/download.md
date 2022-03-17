# Download Datasets

To ease the download of the datasets, we provide a python script that will download the datasets for you. The script is desined to download the dataset in an incremental way. Thus, no worries if you stop the download midway. The script will resume the download where it left off. It is also able to update the dataset if new videos or feature are added. 

## Basic Usage

Here is a simple snippet of code to download the isolated version of the dataset : 

```python
from lsfb_dataset.utils.download.dataset_downloader import DatasetDownloader

destination_folder = './path/to/your/datasets/folder'

ds = DatasetDownloader(destination_folder, dataset="isol")
ds.download()

```
To download the continuous version, just replace `dataset="isol"` with `dataset="cont"`

## Advanced Usage

To give you more control about what is downloaded, the `DatasetDownloader` class propose several parameters : 

- **destination** : Path to the folder where the dataset will be downloaded. (str)
- **dataset** : Type of dataset to download. Must be either `isol` or `cont` (str)
- **exclude** : A list of the data to exclude from the download. The possible values are `video` (to skip the download of the videos), `landmarks` (to skip the download of the mediapipe skeleton information) and `annotations` (to skip the download of gloss annotation for the continuous dataset).  (list)
- **src** : The source url of the dataset. By default, the dataset is downloaded from our server. (str)
- **warning_message** : By default, the script display a warning message to the user. If you want to disable this message, set this parameter to `False`. (bool)


