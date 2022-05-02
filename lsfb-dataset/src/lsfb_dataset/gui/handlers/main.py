from tkinter import filedialog as fd
from ..video_selector import VideoSelector
from os import path
import pandas as pd


class MainHandler:

    def __init__(self, gui):
        self.gui = gui

        self.video_path = None
        self.right_annot_dir = None
        self.left_annot_dir = None
        self.skeleton_dir = None
        self.hands_dir = None

        self.video = None
        self.right_annots = None
        self.left_annots = None
        self.skeleton = None
        self.hands_landmarks = None

    def open_video_file(self):
        file_path = fd.askopenfilename(filetypes=[('Video files', '.mp4 .avi')])
        print(f'Opening video file: [{file_path}].')
        self.gui.video.handler.launch_video(file_path)

    def open_video_selector(self):
        VideoSelector(self.gui)

    def select_video(self, root: str, filename: str):
        print('Select video:', filename)

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

        skeleton = path.join(root, video['upper_skeleton'])
        if path.isfile(skeleton):
            self.skeleton_dir = skeleton
            self.skeleton = pd.read_csv(self.skeleton_dir)
            self.gui.variables['skeleton'].set(True)

        hands = path.join(root, video['hands_cleaned'])
        if path.isfile(hands):
            self.hands_dir = hands
            self.hands_landmarks = pd.read_csv(self.hands_dir)
            self.gui.variables['hands'].set(True)

        video_path = path.join(root, video['relative_path'])
        if path.isfile(video_path):
            self.video_path = video_path
            self.video = video
            self.gui.variables['video_file'].set(True)
            self.gui.video.handler.load_annotations(self.right_annots, self.left_annots)
            self.gui.pred.handler.update_target(self.right_annots, self.video)
            self.gui.video.handler.launch_video(video_path)

    def predict(self, mode):
        model_selection = self.gui.inputs['model_selection']
        model_name = model_selection.get(model_selection.curselection())
        hand_selection = self.gui.inputs['hand_selection']
        hand = hand_selection.get(hand_selection.curselection())

        if mode == 'activity':
            self.gui.pred.handler.predict_activity(model_name, hand, self.skeleton, self.hands_landmarks)
        elif mode == 'signs':
            self.gui.pred.handler.predict_signs(model_name, hand, self.skeleton, self.hands_landmarks)

