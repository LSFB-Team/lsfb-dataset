import pandas as pd
import numpy as np
from cv2 import cv2


def absolute_position(img, pos):
    w = int(img.shape[1])
    h = int(img.shape[0])
    return int(max(0, min(w, w * pos[0]))), int(max(0, min(h, h * pos[1])))


def compute_box(pos, box_size):
    x, y = pos
    x1 = int(max(0, x - box_size / 2))
    y1 = int(max(0, y - box_size / 2))
    x2 = int(x + box_size / 2)
    y2 = int(y + box_size / 2)
    return x1, y1, x2, y2


def extract_box_img(img, box, dim=(256, 256)):
    x1, y1, x2, y2 = box
    return cv2.resize(img[y1:y2, x1:x2], dim, interpolation=cv2.INTER_AREA)


def get_feature_box(img, rel_pos: (float, float), holistic_features: pd.DataFrame, shoulder_length_factor: float):
    abs_pos = absolute_position(img, rel_pos)

    right_shoulder = np.array(absolute_position(img,(
        holistic_features['RIGHT_SHOULDER_X'],
        holistic_features['RIGHT_SHOULDER_Y']
    )))
    left_shoulder = np.array(absolute_position(img, (
        holistic_features['LEFT_SHOULDER_X'],
        holistic_features['LEFT_SHOULDER_Y']
    )))

    base_length = np.sqrt(np.sum(np.square(right_shoulder - left_shoulder)))
    return compute_box(abs_pos, base_length * shoulder_length_factor)
