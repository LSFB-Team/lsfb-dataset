import tkinter as tk
from .handlers.prediction import PredictionHandler


class Prediction(tk.Frame):

    def __init__(self, root=None, **kwargs):
        super(Prediction, self).__init__(root, **kwargs)

        self.handler = PredictionHandler(self)
        self._build()

    def _build(self):
        self.target = tk.Frame(borderwidth=2, relief="groove")
        self.target.pack(fill=tk.Y, pady=5)

        self.pred = tk.Frame(borderwidth=2, relief="groove")
        self.pred.pack(fill=tk.Y, pady=5)
