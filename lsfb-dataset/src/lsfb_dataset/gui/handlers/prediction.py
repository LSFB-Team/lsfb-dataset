from lsfb_dataset.utils.annotations import annotations_to_vec
from lsfb_dataset.visualisation.annotations import create_annot_fig
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import os
from tkinter import messagebox as mb

import torch
import pandas as pd
from lsfb_dataset.models.rnn.rnn_classifier import LSTMClassifier
from lsfb_dataset.utils.numpy import fill_gaps



class PredictionHandler:

    def __init__(self, gui):
        self.gui = gui
        self.target_canvas = None
        self.pred_canvas = None

    def update_target(self, annotations, video):
        if annotations is None:
            return

        if self.target_canvas is not None:
            self.target_canvas.destroy()

        vec = annotations_to_vec(annotations, None, int(video['frames_nb']))
        fig = create_annot_fig(vec)

        canvas = FigureCanvasTkAgg(fig, master=self.gui.target)
        canvas.draw()
        self.target_canvas = canvas.get_tk_widget()
        self.target_canvas.pack(fill=tk.BOTH)

    def update_prediction(self, vec):
        if self.pred_canvas is not None:
            self.pred_canvas.destroy()

        fig = create_annot_fig(vec)

        canvas = FigureCanvasTkAgg(fig, master=self.gui.pred)
        canvas.draw()
        self.pred_canvas = canvas.get_tk_widget()
        self.pred_canvas.pack(fill=tk.BOTH)

    def get_model_filepath(self, model_dir: str, mode: str, model_name: str, hands: str):
        if mode == 'activity':
            mode_prefix = 'ACT'
        elif mode == 'signs':
            mode_prefix = 'SGN'
        else:
            mb.showerror(f'Unknown mode: {mode}.')
            return None

        if model_name == 'LSTM':
            model_prefix = 'LSTM'
        elif model_name == 'EDRN 4D':
            model_prefix = 'EDRN_4D'
        elif model_name == 'Mogrifier LSTM':
            model_prefix = 'mLSTM'
        else:
            mb.showerror(f'Unknown model: {model_name}.')
            return None

        if hands == 'Right':
            hand_prefix = 'RH'
        elif hands == 'Left':
            hand_prefix = 'LH'
        elif hands == 'Both':
            hand_prefix = 'BH'
        else:
            mb.showerror(f'Unknown hands: {hands}.')
            return None

        model_filename = f'{mode_prefix}_{model_prefix}_{hand_prefix}.model'
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.isfile(model_path):
            mb.showerror('Failed to predict', f'Model not found: {model_path}.')
            return None

        return model_path

    def load_features(self, hands, skeleton, hands_landmarks):
        if hands == 'Right':
            hands_landmarks = hands_landmarks.loc[:, hands_landmarks.columns.str.startswith('RIGHT')]
        elif hands == 'Left':
            hands_landmarks = hands_landmarks.loc[:, hands_landmarks.columns.str.startswith('LEFT')]

        return torch.from_numpy(pd.concat([hands_landmarks, skeleton], axis=1).values).float()

    def postprocess_activity(self, vec):
        return fill_gaps(vec, 50, no_gap=1, fill_with=1)

    def predict_activity(self, model_name: str, hands: str, skeleton, hands_landmarks):
        model_dir = r'T:\datasets\lsfb_cont\models'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = self.get_model_filepath(model_dir, 'activity', model_name, hands)
        if model_path is None:
            return
        model = LSTMClassifier(88, 128, 2)
        model.load_state_dict(torch.load(model_path)['best_model'])
        model = model.to(device)
        model.eval()

        features = self.load_features(hands, skeleton, hands_landmarks)
        features = features.to(device)

        with torch.no_grad():
            scores = model(features.unsqueeze(0))
            pred = torch.max(scores, 2)[1]

        vec = pred.squeeze().cpu().numpy()
        vec = self.postprocess_activity(vec)

        self.update_prediction(vec == 1)

    def predict_signs(self, model_name: str, hands: str, skeleton, hands_landmarks):
        model_dir = r'T:\datasets\lsfb_cont\models'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_act_path = self.get_model_filepath(model_dir, 'activity', model_name, hands)
        if model_act_path is None:
            return
        model_act = LSTMClassifier(88, 128, 2)
        model_act.load_state_dict(torch.load(model_act_path)['best_model'])
        model_act = model_act.to(device)
        model_act.eval()

        model_sgn_path = self.get_model_filepath(model_dir, 'signs', model_name, hands)
        if model_sgn_path is None:
            return
        model_sgn = LSTMClassifier(88, 128, 2)
        model_sgn.load_state_dict(torch.load(model_sgn_path)['best_model'])
        model_sgn = model_sgn.to(device)
        model_sgn.eval()

        features = self.load_features(hands, skeleton, hands_landmarks)
        features = features.to(device)

        with torch.no_grad():
            act_scores = model_act(features.unsqueeze(0))
            act_pred = torch.max(act_scores, 2)[1]

        mask = act_pred.squeeze().bool()
        feat_mask = mask.unsqueeze(-1).expand_as(features)
        features = torch.masked_select(features, feat_mask).view(-1, features.size(1))

        with torch.no_grad():
            sgn_scores = model_act(features.unsqueeze(0))
            sgn_pred = torch.max(sgn_scores, 2)[1]

        output = torch.zeros(mask.size(0), dtype=torch.long)
        output[mask.cpu()] = sgn_pred.squeeze().cpu()

        self.update_prediction(output == 1)
