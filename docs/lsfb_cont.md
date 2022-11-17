
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

These very precise data are then aggregated into another set of landmarks, the holistic landmarks. Holistics landmarks are also available in a cleaned version where an interpolation and a smoothing phase have been applied. For more information about those landmarks, you can read the [mediapipe documentation](https://google.github.io/mediapipe/solutions/solutions.html)

