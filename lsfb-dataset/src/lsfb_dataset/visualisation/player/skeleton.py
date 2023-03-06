import numpy as np
import pandas as pd
from .displayable import Displayable
from .cv.landmarks import draw_pose_landmarks, draw_hands_landmarks, draw_face_mesh, draw_face_landmarks


class Skeleton(Displayable):

    def __init__(
            self,
            resolution: tuple[int, int],
            mesh: bool = False,
    ):
        self.resolution = resolution
        self.landmarks_count = 0

        self.pose = None
        self.hands = None
        self.face = None

        self.mesh = mesh

    def add_landmarks(self, landmarks: np.ndarray, landmark_type: str):
        assert landmark_type in ['pose', 'hands', 'face'], 'Unknown landmarks.'
        # landmarks = pd.read_csv(landmarks_path).values
        landmarks = landmarks.reshape((landmarks.shape[0], -1, 2))

        if landmark_type == 'pose':
            self.pose = landmarks
        elif landmark_type == 'hands':
            self.hands = landmarks
        elif landmark_type == 'face':
            self.face = landmarks

        self.landmarks_count = landmarks.shape[0]

    def get_img(self, frame_number: int) -> np.ndarray:
        width, height = self.resolution
        img = np.zeros((height, width, 3), dtype='uint8')
        self.draw(img, frame_number)
        return img

    def draw(self, img: np.ndarray, frame_number: int):
        if self.pose is not None:
            draw_pose_landmarks(img, self.pose[frame_number])

        if self.hands is not None:
            draw_hands_landmarks(img, self.hands[frame_number])

        if self.face is not None:
            if self.mesh:
                draw_face_mesh(img, self.face[frame_number])
            else:
                draw_face_landmarks(img, self.face[frame_number])
