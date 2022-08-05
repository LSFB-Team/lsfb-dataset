from lsfb_dataset.utils.annotations import annotations_to_vec
from lsfb_dataset.visualisation.annotations import create_annot_fig
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import os
from tkinter import messagebox as mb

import torch
import pandas as pd
from lsfb_dataset.models.rnn.rnn_classifier import LSTMClassifier


class PredictionHandler:

    def __init__(self, gui):
        self.gui = gui
        self.target_canvas = None
        self.pred_canvas = None
        self.display_likelihood = False
        self.style = 'black'

        self.current_likelihood = None
        self.current_video = None
        self.current_right_annots = None
        self.current_left_annots = None

        self.use_threshold = False
        self.threshold = 50

    def update_target(self, right_annotations, left_annotations, video):
        if self.target_canvas is not None:
            self.target_canvas.destroy()

        self.current_right_annots = right_annotations
        self.current_left_annots = left_annotations
        self.current_video = video

        vec = annotations_to_vec(right_annotations, left_annotations, int(video['frames_nb']))
        fig = create_annot_fig(vec, style=self.style)

        canvas = FigureCanvasTkAgg(fig, master=self.gui.target)
        canvas.draw()
        self.target_canvas = canvas.get_tk_widget()
        self.target_canvas.pack(fill=tk.BOTH)

    def update_prediction(self, likelihood):
        if self.pred_canvas is not None:
            self.pred_canvas.destroy()

        if self.use_threshold:
            vec = likelihood[:, 1] > (self.threshold/100)
        else:
            vec = torch.max(likelihood, 1)[1] == 1

        self.current_likelihood = likelihood

        fig = create_annot_fig(
            vec,
            style=self.style,
            likelihood=likelihood[:, 1] if self.display_likelihood else None,
            threshold=(self.threshold/100) if self.use_threshold and self.display_likelihood else None
        )

        canvas = FigureCanvasTkAgg(fig, master=self.gui.pred)
        canvas.draw()
        self.pred_canvas = canvas.get_tk_widget()
        self.pred_canvas.pack(fill=tk.BOTH)

    def set_display_likelihood(self, new_val: bool):
        self.display_likelihood = new_val

        if self.current_likelihood is not None:
            self.update_prediction(self.current_likelihood)

    def toggle_threshold(self, new_val: bool):
        self.use_threshold = new_val
        if self.current_likelihood is not None:
            self.update_prediction(self.current_likelihood)

    def update_threshold(self, new_val: int):
        self.threshold = new_val
        if self.current_likelihood is not None and self.use_threshold:
            self.update_prediction(self.current_likelihood)

    def toggle_colored_segmentation(self, new_val: bool):
        if new_val:
            self.style = 'colored'
        else:
            self.style = 'black'

        if self.current_likelihood is not None:
            self.update_prediction(self.current_likelihood)

        if self.current_video is not None:
            self.update_target(self.current_right_annots, self.current_left_annots, self.current_video)

    def get_model_filepath(self, model_dir: str, mode: str, hands: str):
        if mode == 'activity':
            mode_prefix = 'ACT'
        elif mode == 'signs':
            mode_prefix = 'SGN'
        else:
            mb.showerror(f'Unknown mode: {mode}.')
            return None

        if hands == 'right':
            hand_prefix = 'RH'
        elif hands == 'left':
            hand_prefix = 'LH'
        elif hands == 'both':
            hand_prefix = 'BH'
        else:
            mb.showerror(f'Unknown hands: {hands}.')
            return None

        model_filename = f'LSTM_{mode_prefix}_{hand_prefix}.model'
        model_path = os.path.join(model_dir, model_filename)

        return model_path

    def predict(self, models_dir: str, mode: str, hands: str, skeleton_lm, hands_lm):
        model_path = self.get_model_filepath(models_dir, mode, hands)

        if not os.path.isfile(model_path):
            mb.showerror('Model error', f'Model not found: {model_path}')
            return

        num_classes = 2 if mode == 'activity' else 3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMClassifier(130, 128, num_classes, n_layers=2, bidirectional=True)
        model.load_state_dict(torch.load(model_path)['best_model'])
        model = model.to(device)
        model.eval()

        features = torch.from_numpy(pd.concat([hands_lm, skeleton_lm], axis=1).values).float()
        features = features.to(device)

        with torch.no_grad():
            scores = model(features.unsqueeze(0))

        likelihood = torch.sigmoid(scores.squeeze()).detach().cpu()

        self.update_prediction(likelihood)
