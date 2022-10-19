# LSFB_ISOL

The LSFB_ISOL dataset contains 97 000 clips of signs extracted from the [LSFB_CONT](lsfb_cont.md) for 4182 gloss (labels). [Mediapipe](https://mediapipe.dev/) landmarks for the body and the hands are also available for each video. You can download the dataset using the [`DatasetDownloader`](download.md) class.

## Dataset Structure

The dataset contains two folder, one for the video and another one for the features (landmarks). Two `csv` files gives information about the clips and the classes :
 - **clips.csv** : contains one line per clips. Each line contain the label of the sign, the relative path to the features and the associated video.
 - **lemmes.csv** : contains a line for each label. Each line contains the label of the signs and the number of occurence in the dataset.

## Working with LSFB_ISOL


