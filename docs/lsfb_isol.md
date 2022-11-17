# lsfb_isol

The lsfb_isol dataset contains 97 000 clips of signs extracted from the [lsfb_cont dataset](lsfb_cont.md) for 4182 gloss (labels). [Mediapipe](https://mediapipe.dev/) landmarks for the body and the hands are also available for each video. You can download the dataset using the [`DatasetDownloader`](download.md) class.

## Dataset Structure

The dataset contains two folder, one for the video and another one for the features (landmarks). Two `csv` files gives information about the clips and the classes :
 - **clips.csv** : contains one line per clips. Each line contain the label of the sign, the relative path to the features and the associated video.
 - **lemmes.csv** : contains a line for each label. Each line contains the label of the signs its number of occurence in the dataset and a normalized name for the label used as folder name.

The video folder contains a directory for each sign label (named after the label normalized name of the label). Each directory contains all the clips for that particular label.

The features folder also contains a directory for each sign label (named after the label normalized name of the label). The folder contains two subfolder on for the *hands* landmarks and another for the *pose* landmarks. Landmarks are extracted using the mediapipe librairy. For mort information about them you can read the [hands landmarks](https://google.github.io/mediapipe/solutions/hands.html) and [pose landmarks](https://google.github.io/mediapipe/solutions/holistic.html) documentation of mediapipe.


