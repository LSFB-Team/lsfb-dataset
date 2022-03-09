import mediapipe as mp
import numpy as np
from cv2 import cv2
from .shapes import draw_connections, draw_points, draw_line, draw_rect
from ..utils.annotations import get_annotations_in_time_range
from ..utils.holistic_features import get_feature_box, absolute_position, extract_box_img


def draw_feature_box(img, box: (int, int, int, int), color=(0, 255, 0), thickness=1):
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color, thickness=thickness)


def draw_holistic_landmarks(img, features):
    face_pos = (features['FACE_X'], features['FACE_Y'])

    right_hand_pos = (features['RIGHT_HAND_X'], features['RIGHT_HAND_Y'])
    left_hand_pos = (features['LEFT_HAND_X'], features['LEFT_HAND_Y'])

    right_shoulder_pos = (features['RIGHT_SHOULDER_X'], features['RIGHT_SHOULDER_Y'])
    left_shoulder_pos = (features['LEFT_SHOULDER_X'], features['LEFT_SHOULDER_Y'])

    draw_line(img, right_shoulder_pos, left_shoulder_pos)
    draw_points(img, [right_hand_pos, right_shoulder_pos], color=(0, 0, 255))
    draw_points(img, [left_hand_pos, left_shoulder_pos], color=(255, 0, 0))
    draw_points(img, [face_pos], color=(0, 255, 0))


def draw_holistic_boxes(img, features):
    face_pos = (features['FACE_X'], features['FACE_Y'])
    right_hand_pos = (features['RIGHT_HAND_X'], features['RIGHT_HAND_Y'])
    left_hand_pos = (features['LEFT_HAND_X'], features['LEFT_HAND_Y'])

    draw_feature_box(img, get_feature_box(img, right_hand_pos, features, 0.8))
    draw_feature_box(img, get_feature_box(img, left_hand_pos, features, 0.8))
    draw_feature_box(img, get_feature_box(img, face_pos, features, 0.9), color=(0, 0, 255))


def get_holistic_features_img(img, features):
    info_img = np.zeros((256, 3 * 256, 3), dtype='uint8')

    face = features[['FACE_X', 'FACE_Y']].values
    right_hand = features[['RIGHT_HAND_X', 'RIGHT_HAND_Y']].values
    left_hand = features[['LEFT_HAND_X', 'LEFT_HAND_Y']].values
    right_shoulder = features[['RIGHT_SHOULDER_X', 'RIGHT_SHOULDER_Y']].values
    left_shoulder = features[['LEFT_SHOULDER_X', 'LEFT_SHOULDER_Y']].values

    if np.isnan(right_shoulder).sum() or np.isnan(left_shoulder).sum():
        return info_img

    if not np.isnan(right_hand.sum()):
        info_img[:, :256] = extract_box_img(img, get_feature_box(img, right_hand, features, 0.8))

    if not np.isnan(left_hand.sum()):
        info_img[:, 256:512] = extract_box_img(img, get_feature_box(img, left_hand, features, 0.8))

    if not np.isnan(face.sum()):
        info_img[:, 512:] = extract_box_img(img, get_feature_box(img, face, features, 0.9))

    return info_img


def draw_annotation(img, annot, time, index=0):
    x_start = (annot['start'] - time + 3000) / 6000
    x_end = (annot['end'] - time + 3000) / 6000

    draw_rect(img, (x_start, 0.5), (x_end, 1.0), color=(51, 204, 51))

    y_text = 0.4 - (index % 5) * 0.07
    draw_line(img, (x_start, 1), (x_start, y_text - 0.05), color=(51, 204, 51), thickness=1)
    cv2.putText(img, annot['word'], absolute_position(img, (x_start, y_text)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, 2)


def get_annotations_img(df_annot, time, time_offset=3000):
    w = 512
    h = 256

    annot_img = np.zeros((h, w, 3), dtype='uint8')

    df_current_annotations = get_annotations_in_time_range(df_annot, (time - time_offset, time + time_offset))
    for index, annot in df_current_annotations.iterrows():
        draw_annotation(annot_img, annot, time, index=index)
    draw_line(annot_img, (0.5, 0), (0.5, 1.0))

    return annot_img


def draw_pose_landmarks(img, positions):
    connections = mp.solutions.pose.POSE_CONNECTIONS
    draw_connections(img, positions, connections, color=(0, 255, 0))
    draw_points(img, positions)


def draw_hands_landmarks(img, positions):
    draw_hand_landmarks(img, positions[:21], color=(255, 0, 0))
    draw_hand_landmarks(img, positions[21:])


def draw_hand_landmarks(img, positions, color=(0, 0, 255)):
    connections = mp.solutions.hands_connections.HAND_CONNECTIONS
    draw_connections(img, positions, connections, color)


def draw_face_landmarks(img, positions, color=(0, 255, 0)):
    connections = mp.solutions.face_mesh.FACEMESH_CONTOURS
    draw_connections(img, positions, connections, color, thickness=1)
