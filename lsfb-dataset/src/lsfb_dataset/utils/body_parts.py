import numpy as np

LIPS_LANDMARKS_INDICES = (0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
                          267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415)
LEFT_EYE_LANDMARKS_INDICES = (249, 263, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381, 382,
                              384, 385, 386, 387, 388, 390, 398, 466, 474, 475, 476, 477)
RIGHT_EYE_LANDMARKS_INDICES = (7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155, 157, 158,
                               159, 160, 161, 163, 173, 246, 469, 470, 471, 472)
EYES_LANDMARKS_INDICES = tuple(sorted(set(LEFT_EYE_LANDMARKS_INDICES + RIGHT_EYE_LANDMARKS_INDICES)))
UPPER_BODY_LANDMARKS_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def get_associated_landmarks_sets(
        body_parts: tuple[str, ...]
) -> tuple[tuple[str, str | None], ...]:
    lm_sets = tuple()
    for body_part in body_parts:
        if body_part in ['lips', 'left_eye', 'right_eye', 'eyes']:
            lm_sets += (('face', body_part),)
        elif body_part in ['upper_pose']:
            lm_sets += (('pose', body_part),)
        else:
            lm_sets += ((body_part, None),)
    return lm_sets


def get_body_part(
        landmarks: np.ndarray,
        body_part: str,
) -> np.ndarray:
    if body_part == 'lips':
        return landmarks[:, LIPS_LANDMARKS_INDICES]
    elif body_part == 'left_eye':
        return landmarks[:, LEFT_EYE_LANDMARKS_INDICES]
    elif body_part == 'right_eye':
        return landmarks[:, RIGHT_EYE_LANDMARKS_INDICES]
    elif body_part == 'eyes':
        return landmarks[:, EYES_LANDMARKS_INDICES]
    elif body_part == 'upper_pose':
        return landmarks[:, UPPER_BODY_LANDMARKS_INDICES]
    else:
        raise ValueError(f'Unknown body_part: {body_part}.')
