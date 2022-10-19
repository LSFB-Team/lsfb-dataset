import numpy as np
import cv2
from os import path
import random
from typing import Optional
import pandas as pd
from .cv import draw_holistic_landmarks, draw_holistic_boxes, get_holistic_features_img,\
    get_annotations_img, draw_pose_landmarks, draw_hands_landmarks, draw_face_landmarks, draw_face_mesh,\
    get_segmentation_vector_img
from ..utils.helpers import duration_to_str


class VideoPlayer:
    def __init__(self, video_filepath: str):
        if not path.isfile(video_filepath):
            raise FileNotFoundError(f"Video file not found: {video_filepath}")

        self.video_path: str = video_filepath
        self.draw_landmarks: bool = True
        self.draw_boxes: bool = True
        self.draw_duration: bool = True

        self.holistic_features: Optional[pd.DataFrame] = None

        self.right_hand_annotations: Optional[pd.DataFrame] = None
        self.left_hand_annotations: Optional[pd.DataFrame] = None

        self.pose_features: Optional[pd.DataFrame] = None
        self.hands_features: Optional[pd.DataFrame] = None
        self.face_features: Optional[pd.DataFrame] = None

        self.segmentations: list[tuple[str, np.ndarray]] = []

        self.face_meshing = False

        self.only_landmarks = False
        self.only_original = False
        self.crop = None

        self.pause = False
        self.speed = 1.0

    def change_speed(self, speed: float):
        assert 0 < speed, f'Invalid speed value ({speed}).'
        self.speed = speed

    def attach_holistic_features(self, holistic_filepath: str):
        if not path.isfile(holistic_filepath):
            raise FileNotFoundError(
                f"Holistic features file not found: {holistic_filepath}"
            )
        self.holistic_features = pd.read_csv(holistic_filepath)

    def attach_annotations(self, annotations_filepath: str, hand: str):
        if not path.isfile(annotations_filepath):
            raise FileNotFoundError(
                f"Annotations (CSV) file not found: {annotations_filepath}"
            )

        if hand == "right":
            self.right_hand_annotations = pd.read_csv(annotations_filepath)
        elif hand == "left":
            self.left_hand_annotations = pd.read_csv(annotations_filepath)
        else:
            raise ValueError(f"Unknown hand: {hand}")

    def attach_pose_landmarks(self, pose_filepath: str):
        if not path.isfile(pose_filepath):
            raise FileNotFoundError(f"Pose (CSV) file not found: {pose_filepath}")
        self.pose_features = pd.read_csv(pose_filepath)

    def attach_hands_landmarks(self, hands_filepath: str):
        if not path.isfile(hands_filepath):
            raise FileNotFoundError(f"Hands (CSV) file not found: {hands_filepath}")
        self.hands_features = pd.read_csv(hands_filepath)

    def attach_face_landmarks(self, face_landmarks: str):
        if not path.isfile(face_landmarks):
            raise FileNotFoundError(f"Face (CSV) file not found: {face_landmarks}")
        self.face_features = pd.read_csv(face_landmarks)

    def show_landmarks(self, value: bool):
        self.draw_landmarks = value

    def show_boxes(self, value: bool):
        self.draw_boxes = value

    def show_duration(self, value: bool):
        self.draw_duration = value

    def isolate_landmarks(self, value: bool):
        self.only_landmarks = value

    def isolate_original(self, value: bool):
        self.only_original = value

    def play(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_nb = 0
        frame_duration = int(1000 / frame_rate)
        delay = int(frame_duration / self.speed)
        total_duration = int(frame_count * 1000 / frame_rate)
        current_time = 0

        while frame_nb < frame_count and cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            key = cv2.waitKeyEx(delay)
            if key == ord('q'):
                break
            elif key == 32:
                self.pause = True
            elif key == 2424832:
                frame_nb = max(0, frame_nb - 50)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
            elif key == 2555904:
                frame_nb = min(int(frame_count), frame_nb + 50)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)

            while self.pause:
                key = cv2.waitKeyEx(0)
                if key == 32:
                    self.pause = False

            self._show_frame(frame, frame_nb, current_time, total_duration)

            frame_nb += 1
            current_time += frame_duration

        cv2.destroyAllWindows()
        cap.release()

    def get_random_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_nb = random.randint(0, frame_count-1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        success, frame = cap.read()
        if not success:
            raise ValueError('Cannot read the frame.')
        cap.release()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _show_frame(self, frame, current_frame, current_time, total_duration):
        landmarks_frame = None
        original_frame = None

        if self.only_landmarks:
            landmarks_frame = np.zeros(frame.shape, dtype='uint8')
        if self.only_original:
            original_frame = frame.copy()

        self._draw_landmarks(frame, current_frame)
        if landmarks_frame is not None:
            self._draw_landmarks(landmarks_frame, current_frame)

        if self.holistic_features is not None:
            cv2.imshow('Features', get_holistic_features_img(frame, self.holistic_features.iloc[current_frame]))
            if self.draw_boxes:
                draw_holistic_boxes(frame, self.holistic_features.iloc[current_frame])

        if self.right_hand_annotations is not None:
            cv2.imshow(
                "Right hand annotations",
                get_annotations_img(self.right_hand_annotations, current_time),
            )
        if self.left_hand_annotations is not None:
            cv2.imshow(
                "Left hand annotations",
                get_annotations_img(self.left_hand_annotations, current_time),
            )

        for segmentation_name, segmentation in self.segmentations:
            cv2.imshow(
                f"Segmentation - {segmentation_name}",
                get_segmentation_vector_img(segmentation, current_time),
            )

        if landmarks_frame is not None:
            landmarks_frame = self._crop_frame(landmarks_frame)
            cv2.imshow('Landmarks', landmarks_frame)
        if original_frame is not None:
            original_frame = self._crop_frame(original_frame)
            cv2.imshow('Original', original_frame)

        frame = self._crop_frame(frame)
        if self.draw_duration:
            self._show_duration(frame, current_time, total_duration)
        cv2.imshow('Video', frame)
        cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)

    def _draw_landmarks(self, frame, current_frame):
        if self.pose_features is not None:
            draw_pose_landmarks(frame, self.pose_features.iloc[current_frame].values.reshape((-1, 2)))
        if self.hands_features is not None:
            draw_hands_landmarks(frame, self.hands_features.iloc[current_frame].values.reshape((-1, 2)))
        if self.face_features is not None:
            if self.face_meshing:
                draw_face_mesh(frame, self.face_features.iloc[current_frame].values.reshape((-1, 2)))
            else:
                draw_face_landmarks(frame, self.face_features.iloc[current_frame].values.reshape((-1, 2)))
        if self.holistic_features is not None:
            draw_holistic_landmarks(frame, self.holistic_features.iloc[current_frame])

    def _crop_frame(self, frame):
        if self.crop is None:
            return frame

        x1, y1, x2, y2 = self.crop
        return frame[y1:y2, x1:x2]

    def _show_duration(self, frame, current_time: int, total_duration: int):
        duration_info = f'{duration_to_str(current_time)} / {duration_to_str(total_duration)}'
        cv2.putText(frame, duration_info, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, 2)
