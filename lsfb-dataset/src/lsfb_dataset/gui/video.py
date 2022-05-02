import tkinter as tk
from .handlers.video import VideoHandler


class Video(tk.Frame):

    def __init__(self, root, **kwargs):
        super(Video, self).__init__(root, **kwargs)
        self.parent = root
        self._build()
        self.handler = VideoHandler(self)

    def _build(self):
        self.pack_propagate(False)
        self._build_action_bar()

        # self.left_annot_container = tk.Canvas(self, width=250, height=600)
        # self.left_annot_container.pack(side=tk.LEFT, padx=5, anchor=tk.NW)

        self.left_annot = tk.Frame(self)
        self.left_annot.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.Y)
        self.left_true = tk.Canvas(self.left_annot, width=250, height=300)
        self.left_true.pack(side=tk.TOP)
        self.left_pred = tk.Canvas(self.left_annot, width=250, height=300)
        self.left_pred.pack(side=tk.BOTTOM)

        self.container = tk.Canvas(self, width=800, height=600)
        self.container.pack(side=tk.LEFT, padx=15)

        self.right_annot = tk.Frame(self)
        self.right_annot.pack(side=tk.LEFT, anchor=tk.NE, fill=tk.Y)
        self.right_true = tk.Canvas(self.right_annot, width=250, height=300)
        self.right_true.pack(side=tk.TOP)
        self.right_pred = tk.Canvas(self.right_annot, width=250, height=300)
        self.right_pred.pack(side=tk.BOTTOM)

        # self.right_annot_container = tk.Canvas(self, width=250, height=600)
        # self.right_annot_container.pack(side=tk.LEFT, padx=5, anchor=tk.NE)

    def _build_action_bar(self):
        self.bar = tk.Frame(self)
        self.bar.pack(side=tk.BOTTOM, padx=20)

        self.go_forward = tk.Button(self.bar, text="-10")
        self.go_forward.pack(side=tk.LEFT, padx=5, pady=2)

        self.toggle_pause = tk.Button(self.bar, text="Pause", command=lambda: self.handler.toggle_pause())
        self.toggle_pause.pack(side=tk.LEFT, padx=5, pady=2)

        self.go_forward = tk.Button(self.bar, text="+10")
        self.go_forward.pack(side=tk.LEFT, padx=5, pady=2)
