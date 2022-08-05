import numpy as np
from cv2 import cv2
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from lsfb_dataset.visualisation.features import get_annotations_img
from lsfb_dataset.visualisation.features import draw_pose_landmarks, draw_hands_landmarks,\
    draw_face_mesh, draw_face_landmarks


class VideoCapture:

    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError('Could not open the video.')

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.current_frame = 0
        self.current_time = 0

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.delay = (1000 / self.frame_rate)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        if self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.current_frame += 1
                self.current_time += self.delay
                return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return False, None
        else:
            return False, None

    def goto_frame(self, frame_nb: int):
        frame_nb = max(0, frame_nb)
        frame_nb = min(frame_nb, self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        self.current_frame = frame_nb
        self.current_time = frame_nb * self.delay


class VideoHandler:

    def __init__(self, gui):
        self.gui = gui
        self.video_size = (800, 600)

        self.cap = None

        self.photo = None
        self.right_annot_photo = None
        self.left_annot_photo = None

        self.pause = False
        self.delay = 10

        self.right_annots = None
        self.left_annots = None

        self.right_hand = True
        self.left_hand = True

        self.display_video = True
        self.display_skeleton = False
        self.display_hands = False
        self.display_face_mesh = False
        self.display_face_contours = False

        self.post_processing = True

        self.update()

    def launch_video(self, video_path: str):
        self.cap = VideoCapture(video_path)

    def load_annotations(self, right_annots, left_annots):
        self.right_annots = right_annots
        self.left_annots = left_annots

    def toggle_pause(self):
        self.pause = not self.pause

    def toggle_display_video(self, val):
        self.display_video = val

    def toggle_display_skeleton(self, val):
        self.display_skeleton = val

    def toggle_display_hands(self, val):
        self.display_hands = val

    def toggle_display_face_mesh(self, val):
        self.display_face_mesh = val

    def toggle_display_face_contours(self, val):
        self.display_face_contours = val

    def toggle_post_processing(self, val):
        self.post_processing = val
        self.gui.parent.handler.set_post_processing(val)

    def update_hands(self, right_hand: bool, left_hand: bool):
        self.right_hand = right_hand
        self.left_hand = left_hand

        self._clear_annot_img(not right_hand, not left_hand)

    def move(self, offset):
        if self.cap is not None:
            self.cap.goto_frame(self.cap.current_frame + offset)

    def update(self):
        if self.cap is not None and not self.pause:
            success, frame = self.cap.get_frame()

            if not self.display_video:
                frame = np.zeros(shape=frame.shape, dtype='uint8')

            landmarks = self.gui.parent.handler.landmarks

            skeletons = landmarks.get('pose')
            hands = landmarks.get('hands')
            face = landmarks.get('face')

            if self.display_skeleton and skeletons is not None:
                draw_pose_landmarks(frame, skeletons.iloc[self.cap.current_frame].values.reshape((-1, 2)))

            if self.display_hands and hands is not None:
                draw_hands_landmarks(frame, hands.iloc[self.cap.current_frame].values.reshape((-1, 2)))

            if self.display_face_mesh and face is not None:
                draw_face_mesh(frame, face.iloc[self.cap.current_frame].values.reshape((-1, 2)))

            if self.display_face_contours and face is not None:
                draw_face_landmarks(frame, face.iloc[self.cap.current_frame].values.reshape((-1, 2)), (255, 0, 0))

            if success:
                if frame is not None:
                    self.update_video_img(Image.fromarray(frame))
                    if self.right_hand:
                        self.update_right_annotation_img()
                    if self.left_hand:
                        self.update_left_annotation_img()

            else:
                self.cap = None

        self.gui.after(self.delay, self.update)

    def update_video_img(self, img):
        img = ImageOps.contain(img, self.video_size)
        img = ImageOps.pad(img, self.video_size, centering=(.5, .5))
        self.photo = ImageTk.PhotoImage(image=img)
        self.gui.container.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_right_annotation_img(self):
        if self.right_annots is not None:
            img = Image.fromarray(get_annotations_img(self.right_annots, self.cap.current_time, width=250, height=250))
            self.right_annot_photo = ImageTk.PhotoImage(image=img)
            self.gui.right_true.create_image(0, 0, image=self.right_annot_photo, anchor=tk.NW)

    def update_left_annotation_img(self):
        if self.left_annots is not None:
            img = Image.fromarray(get_annotations_img(self.left_annots, self.cap.current_time, width=250, height=250))
            self.left_annot_photo = ImageTk.PhotoImage(image=img)
            self.gui.left_true.create_image(0, 0, image=self.left_annot_photo, anchor=tk.NW)

    def _clear_annot_img(self, right: bool, left: bool):
        if not right and not left:
            return

        empty_img = Image.fromarray(np.full(shape=(250, 250, 3), fill_value=0, dtype='uint8'))
        self.empty_photo = ImageTk.PhotoImage(image=empty_img)
        if right:
            self.gui.right_true.create_image(0, 0, image=self.empty_photo, anchor=tk.NW)
        if left:
            self.gui.left_true.create_image(0, 0, image=self.empty_photo, anchor=tk.NW)
