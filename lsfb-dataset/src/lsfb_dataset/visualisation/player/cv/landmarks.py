from ....datasets.landmark_connections import (
    FULL_BODY_POSE_CONNECTIONS,
    HAND_CONNECTIONS,
    FACEMESH_CONTOURS,
    FACEMESH_TESSELATION,
)
from .utils import draw_connections, draw_points
import numpy as np


def draw_pose_landmarks(img, positions):
    draw_connections(img, positions, FULL_BODY_POSE_CONNECTIONS, color=(0, 255, 0))
    draw_points(img, positions)


def draw_hands_landmarks(img, positions):
    draw_hand_landmarks(img, positions[:21], color=(255, 0, 0))
    draw_hand_landmarks(img, positions[21:])


def draw_hand_landmarks(img, positions, color=(0, 0, 255)):
    draw_connections(img, positions, HAND_CONNECTIONS, color)


def draw_face_landmarks(img, positions, color=(0, 255, 0)):
    draw_connections(img, positions, FACEMESH_CONTOURS, color, thickness=1)
    draw_points(img, positions, color=(0, 0, 255), radius=1)


def draw_face_mesh(img, positions, color=(0, 255, 0)):
    draw_connections(img, positions, FACEMESH_TESSELATION, color, thickness=1)
