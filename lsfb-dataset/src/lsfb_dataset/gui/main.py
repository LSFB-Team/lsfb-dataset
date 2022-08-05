import tkinter as tk
import tkinter.ttk as ttk
from .video import Video
from .prediction import Prediction
from .handlers.main import MainHandler


class MainApp(tk.Tk):

    def __init__(self, root=None):
        super(MainApp, self).__init__(root)
        self._build()
        self.handler = MainHandler(self)

    def _build(self):
        self.title("Main Application")
        self.geometry("1920x1080")
        self.buttons = {}
        self.inputs = {}
        self.infos = {}
        self.variables = {}

        self.menus = tk.Frame(self, width=200)
        self.menus.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_navbar()

        self.video = Video(self, width=1350, height=600, background="#cccccc")
        self.video.pack(side=tk.TOP, padx=5, pady=5)

        self.pred = Prediction(self)
        self.pred.pack(side=tk.BOTTOM, fill=tk.Y, padx=5, pady=5)

        self._build_info_menu()
        self._build_hands_menu()
        self._build_display_menu()
        self._build_prediction_menu()

    def _build_navbar(self):
        self.navbar = tk.Frame(self)
        self.navbar.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        self.buttons['select_video'] = tk.Button(self.navbar, text="Select video",
                                                 command=lambda: self.handler.open_video_selector())
        self.buttons['select_video'].pack(side=tk.LEFT, padx=5)

        # self.buttons['toggle_webcam'] = tk.Button(self.navbar, text="Toggle webcam")
        # self.buttons['toggle_webcam'].pack(side=tk.LEFT, padx=5)

        self.buttons['stop'] = tk.Button(self.navbar, text="Stop")
        self.buttons['stop'].pack(side=tk.LEFT, padx=5)

        self.buttons['load_annotations'] = tk.Button(self.navbar, text="Load annotations")

    def _build_hands_menu(self):
        self.hands_menu = tk.Frame(self.menus, borderwidth=2, relief='ridge')
        self.hands_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NE)

        label = tk.Label(self.hands_menu, text="Hands")
        label.pack(side=tk.TOP)

        self.variables['right_hand'] = tk.BooleanVar(value=True)
        self.inputs['right_hand'] = tk.Checkbutton(
            self.hands_menu, text="Right hand", variable=self.variables['right_hand'],
            command=lambda: self.handler.update_hands())
        self.inputs['right_hand'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['left_hand'] = tk.BooleanVar(value=True)
        self.inputs['left_hand'] = tk.Checkbutton(
            self.hands_menu, text="Left hand", variable=self.variables['left_hand'],
            command=lambda: self.handler.update_hands())
        self.inputs['left_hand'].pack(side=tk.TOP, anchor=tk.NW)

    def _build_info_menu(self):
        self.info_menu = tk.Frame(self.menus, borderwidth=2, relief="ridge")
        self.info_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NE)

        label = tk.Label(self.info_menu, text="Information")
        label.pack(side=tk.TOP)

        self.variables['video_file'] = tk.BooleanVar()
        self.infos['video_file'] = tk.Checkbutton(
            self.info_menu, text="Video file", variable=self.variables['video_file'])
        self.infos['video_file'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['video_file'].config(state='disabled')

        self.variables['right_annot'] = tk.BooleanVar()
        self.infos['right_annot'] = tk.Checkbutton(
            self.info_menu, text="Right hand annotations", variable=self.variables['right_annot'])
        self.infos['right_annot'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['right_annot'].config(state='disabled')

        self.variables['left_annot'] = tk.BooleanVar()
        self.infos['left_annot'] = tk.Checkbutton(
            self.info_menu, text="Left hand annotations", variable=self.variables['left_annot'])
        self.infos['left_annot'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['left_annot'].config(state='disabled')

        self.variables['pose'] = tk.BooleanVar()
        self.infos['pose'] = tk.Checkbutton(
            self.info_menu, text="Skeleton landmarks", variable=self.variables['pose'])
        self.infos['pose'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['pose'].config(state='disabled')

        self.variables['hands'] = tk.BooleanVar()
        self.infos['hands'] = tk.Checkbutton(
            self.info_menu, text="Hands landmarks", variable=self.variables['hands'])
        self.infos['hands'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['hands'].config(state='disabled')

        self.variables['face'] = tk.BooleanVar()
        self.infos['face'] = tk.Checkbutton(
            self.info_menu, text="Face landmarks", variable=self.variables['face'])
        self.infos['face'].pack(side=tk.TOP, anchor=tk.NW)
        self.infos['face'].config(state='disabled')

    def _build_display_menu(self):
        self.display_menu = tk.Frame(self.menus, borderwidth=2, relief="ridge")
        self.display_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NE)

        label = tk.Label(self.display_menu, text="Display")
        label.pack(side=tk.TOP)

        self.variables['display_video'] = tk.BooleanVar(value=True)
        self.inputs['display_video'] = tk.Checkbutton(self.display_menu, text="Display video",
                                                      variable=self.variables['display_video'],
                                                      command=lambda: self.video.handler.toggle_display_video(
                                                          self.variables['display_video'].get()
                                                      ))
        self.inputs['display_video'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['display_skeleton'] = tk.BooleanVar()
        self.inputs['display_skeleton'] = tk.Checkbutton(self.display_menu, text="Display skeleton",
                                                         variable=self.variables['display_skeleton'],
                                                         command=lambda: self.video.handler.toggle_display_skeleton(
                                                             self.variables['display_skeleton'].get()
                                                         ))
        self.inputs['display_skeleton'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['display_hands'] = tk.BooleanVar()
        self.inputs['display_hands'] = tk.Checkbutton(self.display_menu, text="Display hands",
                                                      variable=self.variables['display_hands'],
                                                      command=lambda: self.video.handler.toggle_display_hands(
                                                          self.variables['display_hands'].get()
                                                      ))
        self.inputs['display_hands'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['display_face_mesh'] = tk.BooleanVar()
        self.inputs['display_face_mesh'] = tk.Checkbutton(self.display_menu, text="Display face mesh",
                                                          variable=self.variables['display_face_mesh'],
                                                          command=lambda: self.video.handler.toggle_display_face_mesh(
                                                              self.variables['display_face_mesh'].get()
                                                          ))
        self.inputs['display_face_mesh'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['display_face_contours'] = tk.BooleanVar()
        self.inputs['display_face_contours'] = tk.Checkbutton(
            self.display_menu, text="Display face contours",
            variable=self.variables['display_face_contours'],
            command=lambda: self.video.handler.toggle_display_face_contours(
                self.variables['display_face_contours'].get()
            ))
        self.inputs['display_face_contours'].pack(side=tk.TOP, anchor=tk.NW)

        separator = ttk.Separator(self.display_menu, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)

        self.variables['post_processing'] = tk.BooleanVar(value=True)
        self.inputs['post_processing'] = tk.Checkbutton(self.display_menu, text="Post-processing",
                                                        variable=self.variables['post_processing'],
                                                        command=lambda: self.video.handler.toggle_post_processing(
                                                            self.variables['post_processing'].get()
                                                        ))
        self.inputs['post_processing'].pack(side=tk.TOP, anchor=tk.NW)

        self.variables['colored_segmentation'] = tk.BooleanVar()
        self.inputs['colored_segmentation'] = tk.Checkbutton(
            self.display_menu, text="Colored segmentation",
            variable=self.variables['colored_segmentation'],
            command=lambda: self.pred.handler.toggle_colored_segmentation(
                self.variables['colored_segmentation'].get()
            ))
        self.inputs['colored_segmentation'].pack(side=tk.TOP, anchor=tk.NW)

    def _build_prediction_menu(self):
        self.model_menu = tk.Frame(self.menus, borderwidth=2, relief="ridge")
        self.model_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NE)

        self.variables['enable_treshold'] = tk.BooleanVar()
        self.inputs['enable_treshold'] = tk.Checkbutton(
            self.model_menu,
            text="Prediction threshold",
            variable=self.variables['enable_treshold'],
            command=lambda: self.pred.handler.toggle_threshold(self.variables['enable_treshold'].get())
        )
        self.inputs['enable_treshold'].pack(side=tk.TOP, anchor=tk.NW)

        self.inputs['prediction_thresh'] = tk.Scale(self.model_menu, from_=0, to=100, length=100,
                                                    tickinterval=25, orient=tk.HORIZONTAL)
        self.inputs['prediction_thresh'].set(50)
        self.inputs['prediction_thresh'].pack(side=tk.TOP)
        self.inputs['prediction_thresh'].bind(
            "<ButtonRelease-1>",
            func=lambda _: self.pred.handler.update_threshold(self.inputs['prediction_thresh'].get())
        )

        self.variables['display_likelihood'] = tk.BooleanVar()
        self.inputs['display_likelihood'] = tk.Checkbutton(self.model_menu, text="Display likelihood",
                                                           variable=self.variables['display_likelihood'],
                                                           command=lambda: self.pred.handler.set_display_likelihood(
                                                               self.variables['display_likelihood'].get()
                                                           ))
        self.inputs['display_likelihood'].pack(side=tk.TOP)

        self.buttons['predict_activity'] = tk.Button(self.model_menu, text="Predict activity",
                                                     command=lambda: self.handler.predict('activity'))
        self.buttons['predict_activity'].pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NW)

        self.buttons['predict_signs'] = tk.Button(self.model_menu, text="Predict signs boundaries",
                                                  command=lambda: self.handler.predict('signs'))
        self.buttons['predict_signs'].pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NW)
