import tkinter as tk
from .handlers.video_selector import VideoSelectorHandler


class VideoSelector(tk.Toplevel):

    def __init__(self, root, **kwargs):
        super(VideoSelector, self).__init__(root, **kwargs)
        self.parent = root
        self._build()
        self.handler = VideoSelectorHandler(self)

    def _build(self):
        self.title("Video selector")
        self.geometry("400x300")

        self.root = tk.Button(self, text="Sélectionner le dataset.", command=lambda: self.handler.select_root())
        self.root.pack(side=tk.TOP, anchor=tk.N, pady=5)

        self.list = tk.Listbox(self, selectmode=tk.SINGLE, exportselection=False)
        self.list.pack(side=tk.TOP, fill=tk.X, anchor=tk.NW, pady=10)

        self.select = tk.Button(self, text="Sélectionner la vidéo.", command=lambda: self.handler.select_video())
        self.select.pack(side=tk.TOP, anchor=tk.N)
