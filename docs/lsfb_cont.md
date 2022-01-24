
# LSFB-CONT

LSFB CONT is a dataset for continuous sign language recognition. It is made of videos of deaf people talking together. Each video of the dataset are annotated with the gloss of each sign for each hands. Some video are also directly translated to french. The following section will provide more details about the dataset, its annotation and the various pre-computed features of the video.

In order to download the LSFB CONT dataset, pleas contact me at the address jerome.fink[at]unamur.be


## Videos

The videos of the LSFB CONT dataset are all recorded in 720x576px resolution with 50 FPS. The duration of the video is variable (between 30 seconds and 30 minutes). The signers were asked to accomplish several tasks such as summurizing a video, describing a picture or tell a childhood story.

The *videos.csv* file of the dataset contains the list of all the available video along with several metadata : 

- **filename**: the name of the video file (e.g. `CLSFBI0103_S001_B.mp4`)
- **category**: the category of the video (e.g. `CLSFB - 01 ok`)
- **filepath** : the relative path of the video (eg `videos\CLSFB - 01 ok\CLSFBI0103A_S001_B.mp4`)
- **frames** : the number of frames in the video (ex: `25274`)
- **right_hand_annotations** : the relative path of the right hand annotations
- **left_hand_annotations** : the relative path of the left hand annotations
- **face_positions** : the relative path of the face landmarks
- **hands_positions** : the relative path of the hand landmarks
- **pose_positions** : the relative path of the skeleton landmarks
- **holistic_features** : the relative path of the holistic landmarks
- **holistic_features_cleaned** : the relative path of the cleaned holistic landmarks

The actual content of all these files are presented in the following section.

## Left and Right hands Annotations

The LSFB video are annotated using gloss. Glosses are unique labels associated to each sign. The annotations CSV contains the following columns : 

- **start** : Starting time of the gloss (in milliseconds)
- **end** : Ending time of the gloss (in milliseconds)
- **word** :  The label of the gloss

The annotations CSV files were created based on the original annotations created by the [LSFB-LAB](https://www.unamur.be/lettres/romanes/lsfb-lab). Those annotation are available on demande and use the [ELAN](https://archive.mpi.nl/tla/elan) file format. Along with the gloss annotation, the ELAN file also contains an alligned french translation. However, frenche translation is still a work in progress and will be available in its own CSV file when ready.

## Extracted landmarks

From each frame of each video a set of landmarks is extracted. These landmarks relate to the main speaker in the video. The different sets of landmarks extracted:

- The face
- The pose (skeleton)
- The hands

These very precise data are then aggregated into another set of landmarks, the holistic landmarks. Holistics landmarks are also available in a cleaned version where an interpolation and a smoothing phase have been applied.

### Face landmarks

There are 468 facial landmarks (numbered from 0 to 467).
The corresponding CSV file therefore has 2x468 (for X and Y respectively) or 936 columns.
Let `FACE_0_X, FACE_0_Y, FACE_1_X, FACE_1_Y, ... , FACE_467_X, FACE_467_Y,`.

<img src="https://mediapipe.dev/assets/img/photos/faceMesh.jpg" width="500px" alt="Face landmarks"/>

### Skeleton landmarks

The pose landmarks are 33 in number (numbered 0 to 32).
The corresponding CSV file therefore has 2x33 (for X and Y respectively) or 66 columns.
In order to identify the part of the body concerned, please refer to the image below.
The columns are in the same order as in the image and are in capitals.
So: `NOSE_X, NOSE_Y, LEFT_EYE_INNER_X, LEFT_EYE_INNER_Y, ..., RIGHT_FOOT_INDEX_X, RIGHT_FOOT_INDEX_Y`.

<img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png" width="500px" alt="Pose landmarks"/>

### Hand landmarks

The hand landmarks are primarily for the right hand, and then the left hand.
Each hand has 21 landmarks (numbered from 0 to 20). Each landmark consists of an X position and a Y position.
This makes a total of (2 hands) x (21 landmarks) x (2 dimensions), or 84 columns.

The column names are in upper case and are ordered first by hand, then by the name of the landmark in the image below, and finally by X and Y.
This gives: `RIGHT_WRIST_X, RIGHT_WRIST_Y, ..., RIGHT_PINKY_TIP_X, RIGHT_PINKY_TIP_Y, LEFT_WRIST_X, LEFT_WRIST_Y, ..., LEFT_PINKY_TIP_X, LEFT_PINKY_TIP_Y`

<img src="https://google.github.io/mediapipe/images/mobile/hand_landmarks.png" width="500px" alt="Hand landmarks"/>

### Holistic landmarks

There are 5 holistic landmarks:

1. the **face** (FACE): A point located at the average postion of the face landmarks
2. the right hand (RIGHT_HAND): A point located at the average of the right hand landmarks
3. left hand (LEFT_HAND): A point located at the average of the left hand landmarks
4. right shoulder (RIGHT_SHOULDER): A point located at the right shoulder marker
5. the left shoulder (LEFT_SHOULDER): A point located at the left shoulder marker

These landmarks are separated into two dimensions (X and Y).
So there are 10 columns in the CSV file: `FACE_X`,`FACE_Y`,`RIGHT_HAND_X`,`RIGHT_HAND_Y`,`LEFT_HAND_X`,`LEFT_HAND_Y`,`RIGHT_SHOULDER_X`,`RIGHT_SHOULDER_Y`,`LEFT_SHOULDER_X`,`LEFT_SHOULDER_Y`

