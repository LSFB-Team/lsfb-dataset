from cv2 import cv2
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
import pandas as pd
from lsfb_dataset.visualisation.features import get_annotations_img
from lsfb_dataset.visualisation.features import draw_pose_landmarks, draw_hands_landmarks


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

        self.display_video = True
        self.display_skeleton = False
        self.display_hands = False

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

    def update(self):
        if self.cap is not None and not self.pause:
            success, frame = self.cap.get_frame()

            skeletons = self.gui.parent.handler.skeleton
            hands = self.gui.parent.handler.hands_landmarks

            if self.display_skeleton and skeletons is not None:
                draw_pose_landmarks(frame, skeletons.iloc[self.cap.current_frame].values.reshape((-1, 2)))

            if self.display_hands and hands is not None:
                draw_hands_landmarks(frame, hands.iloc[self.cap.current_frame].values.reshape((-1, 2)))

            if success:
                if frame is not None:
                    self.update_video_img(Image.fromarray(frame))
                    self.update_right_annotation_img()
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
            img = Image.fromarray(get_annotations_img(self.right_annots, self.cap.current_time, width=250, height=300))
            self.right_annot_photo = ImageTk.PhotoImage(image=img)
            self.gui.right_true.create_image(0, 0, image=self.right_annot_photo, anchor=tk.NW)

    def update_left_annotation_img(self):
        if self.left_annots is not None:
            img = Image.fromarray(get_annotations_img(self.left_annots, self.cap.current_time, width=250, height=300))
            self.left_annot_photo = ImageTk.PhotoImage(image=img)
            self.gui.left_true.create_image(0, 0, image=self.left_annot_photo, anchor=tk.NW)
