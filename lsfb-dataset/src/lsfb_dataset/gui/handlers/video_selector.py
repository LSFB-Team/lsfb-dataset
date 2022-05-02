from tkinter import filedialog as fd, messagebox as mb
import pandas as pd
from os import path


class VideoSelectorHandler:

    def __init__(self, gui):
        self.gui = gui
        self.root = None
        self.videos = None

    def select_root(self):
        self.root = fd.askdirectory()
        videos_path = path.join(self.root, 'videos.csv')

        if not path.isfile(videos_path):
            mb.showerror('Videos not found', 'Could not find the video.csv file.')
            return

        self.videos = pd.read_csv(videos_path)
        self.update_video_list()

    def update_video_list(self):
        listbox = self.gui.list

        for item in listbox.children:
            item.destroy()

        for idx, filename in enumerate(self.videos['filename']):
            listbox.insert(idx+1, filename)

        self.gui.focus_set()

    def select_video(self):
        listbox = self.gui.list

        filename = listbox.get(listbox.curselection())
        self.gui.parent.handler.select_video(self.root, filename)
        self.gui.destroy()
