
# LSFB-CONT

LSFB CONT is a dataset for continuous sign language recognition. It is made of videos of deaf people talking together. Each video of the dataset are annotated with the gloss of each sign for each hands. Some video are also directly translated to french. The following section will provide more details about the dataset, its annotation and the various pre-computed features of the video.

In order to download the LSFB CONT dataset, pleas contact me at the address jerome.fink[at]unamur.be


## Instances

The signers were asked to accomplish several tasks such as summarizing a video, describing a picture or tell a childhood story.
All those sign language discussion resulted in two instances corresponding o the videos of the two signers.
The *instances.csv* file of the dataset contains the list of all the available instances in the dataset: 

- **id**: the ID of the instance (e.g. `CLSFBI0103_S001_B`)
- **signer_id**: the category of the video (e.g. `001`)
- **session_id** : the session of the video (e.g. `01`)
- **task_id** : the category of the video (e.g. `03`)
- **n_frames** : the number of frames in the video (e.g. `25264`)
- **n_signs** : the number of signs in the video (e.g. `398`)

## Videos

The videos of the LSFB CONT dataset are all recorded in 720x576px resolution with 50 FPS.
The duration of the video is variable (between 30 seconds and 30 minutes).

Each video can be retrieved from the ID of the corresponding instance.
For example: `./videos/CLSFBI0103_S001_B.mp4` for the instance `CLSFBI0103_S001_B`.

## Annotations

The LSFB video are annotated using gloss. Glosses are unique labels associated to each sign.
All the annotations are contained in a single file where each instance ID is associated to a list of annotations `(start, end, label)` where:
* `start`: starting time of the sign (milliseconds)
* `end`: ending time of the sign (milliseconds)
* `label`: the label of the sign (gloss)

There are multiple annotations files:
* `./annotations/signs_left_hand.json`: all the signs performed with the left hand;
* `./annotations/signs_right_hand.json`: all the signs performed with the right hand;
* `./annotations/signs_both_hands.json`: all the signs performed with any hand;

The annotations files were created based on the original annotations created by the [LSFB-LAB](https://www.unamur.be/lettres/romanes/lsfb-lab). Those annotation are available on demande and use the [ELAN](https://archive.mpi.nl/tla/elan) file format. Along with the gloss annotation, the ELAN file also contains an alligned french translation. However, frenche translation is still a work in progress and will be available in its own CSV file when ready.

## Poses

From each frame of each video a set of landmarks is extracted using [MediaPipe](https://developers.google.com/mediapipe).
These landmarks relate to the main speaker in the video.
The different sets of landmarks extracted:

* The face: 468 landmarks;
* The pose (body): 30 landmarks;
* The left hand: 21 landmarks;
* The right hand: 21 landmarks.

Each video can be retrieved from the ID of the corresponding instance.
For example, the landmarks of the left hand of the signer of the instance `CLSFBI0103_S001_B`
are located in the file `./poses/left_hand/CLSFBI0103_S001_B.npy`.
You can then load this file using [Numpy](https://numpy.org/).

You can also load the raw landmarks in the `poses_raw` folder.
