from tkinter import filedialog as fd
from tkinter import messagebox as mb
from ..video_selector import VideoSelector
from os import path
import pandas as pd


class MainHandler:

    def __init__(self, gui):
        self.gui = gui

        self.video_path = None
        self.right_annot_dir = None
        self.left_annot_dir = None

        self.root = None
        self.video = None
        self.right_annots = None
        self.left_annots = None

        self.landmark_directories = {}
        self.landmarks = {}

    def open_video_file(self):
        file_path = fd.askopenfilename(filetypes=[('Video files', '.mp4 .avi')])
        print(f'Opening video file: [{file_path}].')
        self.gui.video.handler.launch_video(file_path)

    def open_video_selector(self):
        VideoSelector(self.gui)

    def select_video(self, root: str, filename: str):
        print('Select video:', filename)

        self.root = root

        videos = pd.read_csv(path.join(root, 'videos.csv'))
        video = videos.loc[videos['filename'] == filename, :].iloc[0, :]

        r_annot = path.join(root, video['right_hand_annotations'])
        if path.isfile(r_annot):
            self.right_annot_dir = r_annot
            self.right_annots = pd.read_csv(r_annot)
            self.gui.variables['right_annot'].set(True)

        l_annot = path.join(root, video['left_hand_annotations'])
        if path.isfile(l_annot):
            self.left_annot_dir = l_annot
            self.left_annots = pd.read_csv(l_annot)
            self.gui.variables['left_annot'].set(True)

        post_processing = self.gui.variables['post_processing'].get()

        self.update_landmarks(root, video, 'pose', post_processing, show_error=False)
        self.update_landmarks(root, video, 'hands', post_processing, show_error=False)
        self.update_landmarks(root, video, 'face', post_processing, show_error=False)

        video_path = path.join(root, video['relative_path'])
        if path.isfile(video_path):
            self.video_path = video_path
            self.video = video
            self.gui.variables['video_file'].set(True)
            self.gui.video.handler.load_annotations(self.right_annots, self.left_annots)
            self.update_target()
            self.gui.video.handler.launch_video(video_path)

    def update_target(self):
        right_hand = self.gui.variables['right_hand'].get()
        left_hand = self.gui.variables['left_hand'].get()
        right_annots = None
        left_annots = None
        if right_hand:
            right_annots = self.right_annots
        if left_hand:
            left_annots = self.left_annots
        self.gui.pred.handler.update_target(right_annots, left_annots, self.video)

    def update_hands(self):
        right_hand = self.gui.variables['right_hand'].get()
        left_hand = self.gui.variables['left_hand'].get()

        self.gui.video.handler.update_hands(right_hand, left_hand)
        self.update_target()

    def update_landmarks(self, root: str, video, lm_type: str, post_process: bool, show_error=True):
        filename = video['filename']
        category = video['category']

        if not post_process:
            lm_dir = path.join(root, 'features', 'landmarks', category, filename, f'{lm_type}.csv')
        else:
            if lm_type == 'face':
                feature = 'face_cleaned'
            elif lm_type == 'hands':
                feature = 'hands_cleaned'
            elif lm_type == 'pose':
                feature = 'upper_skeleton'
            else:
                raise ValueError('Unknown post-processed landmarks: %s' % lm_type)
            lm_dir = path.join(root, video[feature])

        if not path.isfile(lm_dir):
            if show_error:
                mb.showerror('Landmarks not found', f'Could not find landmarks: {lm_dir}')
            self.landmark_directories[lm_type] = None
            self.landmarks[lm_type] = None
            self.gui.variables[lm_type].set(False)
        else:
            self.landmark_directories[lm_type] = lm_dir
            self.landmarks[lm_type] = pd.read_csv(lm_dir)
            self.gui.variables[lm_type].set(True)

    def set_post_processing(self, new_value):
        if self.root is None:
            return
        if self.video is None:
            return

        self.update_landmarks(self.root, self.video, 'pose', new_value, show_error=True)
        self.update_landmarks(self.root, self.video, 'hands', new_value, show_error=True)
        self.update_landmarks(self.root, self.video, 'face', new_value, show_error=True)

    def predict(self, mode):
        if self.root is None:
            return

        if not self.gui.variables['post_processing'].get():
            mb.showerror('Prediction failed', 'Post-processing step is mandatory. Enable it and retry.')
            return

        right_hand = self.gui.variables['right_hand'].get()
        left_hand = self.gui.variables['left_hand'].get()

        if right_hand and left_hand:
            hands = 'both'
        elif right_hand:
            hands = 'right'
        elif left_hand:
            hands = 'left'
        else:
            mb.showerror('Missing option', 'You have to select at least one hand.')
            return

        if not mb.askokcancel('Prediction confirmation', f'Prediction: {mode} ; Hands: {hands}'):
            return

        models_dir = path.join(self.root, 'models')
        self.gui.pred.handler.predict(models_dir, mode, hands, self.landmarks['pose'], self.landmarks['hands'])

