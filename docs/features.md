
# LSFB CONT - Features

## Videos with annotations

First, a set of video files is provided.
Each of these videos is several minutes long and has a category and an associated file containing
annotations in sign language (ELAN file).

For example, the video file `CLSFB - 01 ok/CLSFBI0103_S001_B.mp4` has `CLSFB - 01 ok` as a category
and an associated ELAN annotation file `CLSFBI0103.eaf`.

## The annotation list

As explained above, each video has associated annotations. However, in addition to the ELAN file,
a CSV file with the list of annotations and their start and end is provided. The annotations are
separated according to the hand used.

For example, `features/annotations/CLSFB - 01 ok/CLSFBI0103A_S001_B.mp4/right_hand.csv` is an annotation file for the right hand of the video
for the right hand of the video `CLSFBI0103A_S001_B.mp4`.

### Columns in an annotation file (CSV)

1. **start**: the start of the annotation (in milliseconds)
2. **end** : the end of the annotation (in milliseconds)
3. **word**: the annotation, e.g. `HOUSE(1X)`

## Extracted landmarks

From each frame of each video a set of landmarks is extracted.
These landmarks relate to the main speaker in the video. Here are the different sets of landmarks extracted:

- The face
- The pose (skeleton)
- The hands

This very precise data is then aggregated into another set of landmarks, the holistic landmarks.
These are also offered in a cleaned version where an interpolation phase and a smoothing phase have been performed.

For example, the video `CLSFB - 01 ok/CLSFBI0103_S001_B.mp4` has several associated landmarks
in the folder `features/landmarks/CLSFB - 01 ok/CLSFBI0103A_S001_B.mp4` :

- `face.csv` : the face landmarks
- `pose.csv` : pose, skeleton landmarks
- `hands.csv`: hand landmarks
- `holistic.csv`: holistic landmarks
- `holistic_cleaned.csv` : holistic cleaned landmarks

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

1. the face (FACE): the average of the face landmarks
2. the right hand (RIGHT_HAND): the average of the right hand landmarks
3. left hand (LEFT_HAND): the average of the left hand landmarks
4. right shoulder (RIGHT_SHOULDER): the right shoulder marker
5. the left shoulder (LEFT_SHOULDER): the left shoulder marker

These landmarks are separated into two dimensions (X and Y).
So there are 10 columns in the CSV file: `FACE_X,FACE_Y,RIGHT_HAND_X,RIGHT_HAND_Y,LEFT_HAND_X,LEFT_HAND_Y,RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y,LEFT_SHOULDER_X,LEFT_SHOULDER_Y

## A file that links videos to their features

In addition to the features discussed above.
A `videos.csv` file is a list of videos with information about them and paths to their features.

Each row is for one video. Here are the different columns in this file:

- **filename**: the name of the video file (e.g. `CLSFBI0103_S001_B.mp4`)
- **category**: the category of the video (e.g. `CLSFB - 01 ok`)
- filepath : the relative path of the video (eg `videos\CLSFB - 01 ok\CLSFBI0103A_S001_B.mp4`)
- frames : the number of frames in the video (ex: `25274`)
- right_hand_annotations : the relative path of the right hand annotations
- left_hand_annotations : the relative path of the left hand annotations
- face_positions : the relative path of the face landmarks
- hands_positions: the relative path of the hand landmarks
- pose_positions : the relative path of the skeleton landmarks
- holistic_features : the relative path of the holistic landmarks
- holistic_features_cleaned : the relative path of the cleaned holistic landmarks

The two first columns are mandatory. The others can be empty.
