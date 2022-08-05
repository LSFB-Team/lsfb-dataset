import tkinter as tk
from tkinter import Label, filedialog as fd
import cv2
from PIL import Image, ImageTk, ImageOps
import time
import numpy as np


class SignSegmentor(tk.Tk):

    def __init__(self, root=None):
        super(SignSegmentor, self).__init__(root)

        self.cap = None

        self.video_size = (800, 600)
        self.current_frame = 0
        self.total_frame = None
        self.pause = False
        self.goto_frame = None

        self._build()
    
    def _build(self):
        self.title("Sign Segmentor")
        self.geometry("1600x900")

        self._build_top_menu()
        self._build_video_player()
        self._build_right_menu()

        self._update_video_frame()
    
    def _build_top_menu(self):
        frame_top_menu = tk.Frame(self, )
        frame_top_menu.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        btn_load_video = tk.Button(frame_top_menu, text="Load video", command=lambda: self._open_video_file())
        btn_load_video.pack(side=tk.LEFT, padx=5)

        btn_webcam = tk.Button(frame_top_menu, text="Webcam", command=lambda: self._open_webcam())
        btn_webcam.pack(side=tk.LEFT, padx=5)

        btn_stop = tk.Button(frame_top_menu, text="Stop", command=lambda: self._stop_video())
        btn_stop.pack(side=tk.LEFT, padx=5)
    
    def _build_right_menu(self):
        frame_right_menu = tk.Frame(self, borderwidth=2, relief="ridge")
        frame_right_menu.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        model_list_label = Label(frame_right_menu, text="Models")
        model_list_label.pack(side=tk.TOP)

        model_list = tk.Listbox(frame_right_menu)
        model_list.insert(1, 'LSTM')
        model_list.insert(2, 'EDRN 2D')
        model_list.insert(3, 'EDRN 4D')
        model_list.insert(4, 'Mogrifier LSTM')
        model_list.pack(side=tk.TOP)

        model_threshold_label = Label(frame_right_menu, text="Prediction threshold")
        model_threshold_label.pack(side=tk.TOP)

        model_threshold = tk.Scale(frame_right_menu, from_=0, to=100, length=100, tickinterval=25, orient=tk.HORIZONTAL)
        model_threshold.set(50)
        model_threshold.pack(side=tk.TOP)

        display_likelihood = tk.Checkbutton(frame_right_menu, text="Display likelihood")
        display_likelihood.pack(side=tk.TOP)

        display_skeleton = tk.Checkbutton(frame_right_menu, text="Display skeleton")
        display_skeleton.pack(side=tk.TOP)
    
    def _build_video_player(self):
        frame_video = tk.Frame(self, width=800, height=600, background="#cccccc")
        frame_video.pack(side=tk.LEFT, padx=5, pady=5)

        self.label_video = tk.Label(frame_video)
        self.label_video.pack()

        self._build_video_player_controls(frame_video)
    
    def _build_video_player_controls(self, frame_video):
        frame_video_controls = tk.Frame(frame_video)
        frame_video_controls.pack(side=tk.BOTTOM)

        btn_go_left = tk.Button(frame_video_controls, text="<--", command=lambda: self._goto_frame(self.current_frame - 500))
        btn_go_left.bind_all('<Left>', lambda _: self._goto_frame(self.current_frame - 50))
        btn_go_left.pack(side=tk.LEFT)

        btn_pause = tk.Button(frame_video_controls, text="Pause", command=lambda: self._toggle_pause())
        btn_pause.pack(side=tk.LEFT)
        btn_pause.bind_all('<KeyPress-space>', lambda _: self._toggle_pause())

        btn_go_right = tk.Button(frame_video_controls, text="-->", command=lambda: self._goto_frame(self.current_frame + 500))
        btn_go_right.bind_all('<Right>', lambda _: self._goto_frame(self.current_frame + 50))
        btn_go_right.pack(side=tk.LEFT)

        self.slider_video = tk.Scale(frame_video_controls, orient=tk.HORIZONTAL)
        self.slider_video.pack(fill=tk.X)
        self.slider_video.bind('<ButtonRelease-1>', lambda evnt: self._goto_frame(evnt.widget.get()*50))
        self._update_video_slider()
    
    def _open_video_file(self):
        file_path = fd.askopenfilename(filetypes=[('Video files', '.mp4 .avi')])
        print(f'Opening video file: [{file_path}].')
        self._update_capture(file_path)
        self.current_frame = 0
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pause = False
        self._update_video_slider(disabled=False, frames_nb=self.total_frame)
    
    def _open_webcam(self):
        print('Opening webcam.')
        self._update_capture(0)
        self._update_video_slider()
        self.current_frame = 0
        self.total_frame = None
        self.pause = False
    
    def _stop_video(self):
        print('Stop current video.')
        self._update_capture(None)
        self._update_video_slider()
    
    def _goto_frame(self, frame):
        if self.cap is None:
            return
        if self.total_frame is None:
            return
        self.goto_frame = min(max(0, frame), self.total_frame)
    
    def _toggle_pause(self):
        self.pause = not self.pause
    
    def _update_capture(self, source):
        if self.cap is not None:
            self.cap.release()
        
        if source is None:
            self.cap = None
            return
        
        self.cap = cv2.VideoCapture(source)
    
    def _update_video_slider(self, disabled=True, frames_nb=None):
        steps = 0
        state = tk.DISABLED

        if frames_nb is not None:
            steps = frames_nb // 50
        if not disabled:
            state = tk.ACTIVE

        self.slider_video.config(state=state, takefocus=0, from_=0, to=steps, length=1000, tickinterval=60)

    def _update_video_frame(self):
        if self.cap is None:
            self._display_cv_img(np.zeros((self.video_size[1], self.video_size[0], 3), dtype='uint8'))
            self.label_video.after(200, lambda: self._update_video_frame())
            return
        
        if self.goto_frame is not None:
            self.current_frame = self.goto_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.goto_frame)
            self.goto_frame = None
        
        if self.pause:
            self.label_video.after(200, lambda: self._update_video_frame())
            return
        
        if self.total_frame is not None and self.current_frame >= self.total_frame:
            self.label_video.after(200, lambda: self._update_video_frame())
            return
        
        start_time = time.time() * 1000.0
        success, cv_img = self.cap.read()

        if not success:
            self.cap = None
            print('Failed to read video stream.')
        
        self._display_cv_img(cv_img)
        self.current_frame += 1
        
        end_time = time.time() * 1000.0
        self.label_video.after(max(1, 20 - int(end_time - start_time)), lambda: self._update_video_frame())
    
    def _display_cv_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageOps.contain(img, self.video_size)
        img = ImageOps.pad(img, self.video_size, centering=(.5, .5))
        img = ImageTk.PhotoImage(image=img)

        self.label_video.img = img
        self.label_video.configure(image=img)


if __name__ == '__main__':
    app = SignSegmentor()
    app.mainloop()
