import mediapipe as mp
from cv2 import cv2


class HolisticDetector:
    """
    The holistic detector detect and track shoulders and hands from openCV frames.

    Parameters:
        min_detection_confidence (float) : threshold of detection for the model (between 0 and 1)
        min_track_confidence (float) : threshold of tracking for the model (between 0 and 1)
        model_complexity (int) : indicates the level of complexity of the model (0, 1 or 2)
                                 An higher value means a slower but more efficient model
    """

    def __init__(self,
                 min_detection_confidence=0.2,
                 min_track_confidence=0.1,
                 model_complexity=2):
        self.detection_confidence = min_detection_confidence
        self.track_confidence = min_track_confidence
        self.model_complexity = model_complexity

        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def detect(self, img, draw=True):
        """
        Detect the shoulders and hands from an openCV image. This must be called before fetching the results.

        Parameters:
            img (openCV BGR frame) : The openCV frame from which the shoulders and hands are detected.
            draw (bool) : If true, draws the detected shoulders and hands on the source image.
        """

        with self._get_mp_detector() as detector:
            results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            left_hand_style = self.mp_styles.DrawingSpec(color=(255, 0, 0))
            right_hand_style = self.mp_styles.DrawingSpec(color=(0, 0, 255))

            w = img.shape[1]
            h = img.shape[0]

            if results.pose_landmarks:
                left_shoulder = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]

                left_position = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                right_position = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                if draw:
                    self._draw_shoulders(img, left_position, right_position)

            if results.left_hand_landmarks:
                if draw:
                    self._draw_hand(img, results.left_hand_landmarks, left_hand_style)

            if results.right_hand_landmarks:
                if draw:
                    self._draw_hand(img, results.right_hand_landmarks, right_hand_style)

    def _get_mp_detector(self):
        return self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.track_confidence,
            model_complexity=self.model_complexity,
            enable_segmentation=True,
            smooth_segmentation=True
        )

    def _draw_hand(self, img, landmarks, style):
        self.mp_draw.draw_landmarks(img, landmarks, self.mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=style)

    def _draw_shoulders(self, img, left_position, right_position):
        cv2.line(img, left_position, right_position, (255, 255, 255), 2)
        cv2.circle(img, left_position, 3, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, right_position, 3, (0, 0, 255), cv2.FILLED)
