import cv2
import numpy as np
from PyQt5 import QtCore

class Camera(QtCore.QThread):
    signal_frame_changed = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_index, framerate, gamma):
        QtCore.QThread.__init__(self)
        self.running = False
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)

        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        self.cap.set(cv2.CAP_PROP_GAMMA, gamma)
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -2)



    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.signal_frame_changed.emit(frame)
                print(self.cap.get(cv2.CAP_PROP_FPS))
        self.cap.release()

# class VideoWriter(QtCore.QThread):
#     def __init__(self, q):
#         self.running = False
#         self.q = q
#
#     def run(self):
#         savepath = "~/testvideo.avi"
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         out = cv2.VideoWriter(savepath, fourcc, framerate, (w, h), isColor=False)
#         while self.running:
#             frame = self.q.get
#
