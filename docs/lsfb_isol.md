# LSFB ISOL

The lsfb_isol dataset contains 120,000 clips of signs extracted from the [lsfb_cont dataset](lsfb_cont.md) for more than 4000 different signs (gloss/labels).
[Mediapipe](https://developers.google.com/mediapipe) poses are also available for each video.
You can download the dataset using the [`Downloader`](download.md) class.

## Dataset Structure

The structure of the dataset is very similar of the LSFB CONT one.

### Instances

The `instances.csv` file contains all the instances of the dataset:
* id: The id of the instance (e.g. `CLSFBI0103A_S001_B_286114_286297`)
* sign: The gloss of the sign corresponding to the instance (e.g. `MAISON`)
* signer: The signer that performed the sign (e.g. `S001`)
* start: The starting time of the sign in LSFB CONT (in milliseconds)
* end: The ending time of the sign in LSFB CONT (in milliseconds)

### Video Clips

You can easily retrieves the video clip corresponding to any instance.
For example, the video of the instance `CLSFBI0103A_S001_B_286114_286297` is located at `./videos/CLSFBI0103A_S001_B_286114_286297.mp4`.

### Poses

From each frame of each video a set of landmarks is extracted using [MediaPipe](https://developers.google.com/mediapipe).
These landmarks relate to the main speaker in the video.
The different sets of landmarks extracted:

* The face: 468 landmarks;
* The pose (body): 30 landmarks;
* The left hand: 21 landmarks;
* The right hand: 21 landmarks.

Each video can be retrieved from the ID of the corresponding instance.
For example, the landmarks of the left hand of the signer of the instance `CLSFBI0103A_S001_B_286114_286297`
are located in the file `./poses/left_hand/CLSFBI0103A_S001_B_286114_286297.npy`.
You can then load this file using [Numpy](https://numpy.org/).

You can also load the raw landmarks in the `poses_raw` folder.
