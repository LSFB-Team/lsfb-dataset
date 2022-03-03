# Visualisation

The visualisation module allows you to visualise the video present in the dataset along with all the metadata associated with them (landmarks, labels).

The main class of this module is the `VideoPlayer` class : 

```python
from lsfb_dataset.visualisation.video import VideoPlayer

video_path = "./path/to/video.mp4"

player = VideoPlayer(video_path)
player.play()

``` 

The `VideoPlayer` accept the path (`str`) to the video as an argument. The `play` function is then called to display the video selected using OpenCV.

The class also provides methods to attach metadata to the video :


```python
from lsfb_dataset.visualisation.video import VideoPlayer

# Creating the VideoPlayer for the video
player = VideoPlayer(vid_path)

# Attaching the csv file to the video
player.attach_holistic_features(hollistic_features_path)
player.attach_annotations(right_hand_path, hand='right')
player.attach_annotations(left_hand_path, hand='left')

# Telling the player that those information should be displayed
player.show_landmarks(True)
player.show_boxes(True)
player.show_duration(True)

player.play()

``` 

In this more complex example, we attach the csv file containing the corresponding metadata to the video and we tell the player which additional information should be displayed.