from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtMultimedia import QCameraInfo
from qt5_camera.ui.camera_controls_ui import Ui_TileMapWidget as camera_control_ui
import sys
from queue import Queue
from qt5_camera.Camera import Camera, Camera_Rec
import numpy as np
import cv2
import time
from qt5_camera.VideoWriter import Writer, Timer

class CameraWidget(QtWidgets.QDockWidget,camera_control_ui):
    def __init__(self):
        QtWidgets.QDockWidget.__init__(self)
        camera_control_ui.__init__(self)
        self.setWindowTitle("Camera Controls")
        self.ui = camera_control_ui()
        self.ui.setupUi(self)
        self.connect_ui()

        self.get_available_cameras()
        self.connect_camera()

    def connect_ui(self):
        self.ui.buttonScanCameras.clicked.connect(self.get_available_cameras)
        self.ui.buttonConnectCamera.clicked.connect(self.connect_camera)
        self.ui.buttonConnectCamera.setVisible(False)
        self.ui.comboBoxCameras.currentTextChanged.connect(self.connect_camera)
        self.ui.checkBoxViewCamera.toggled.connect(self.toggle_view_camera)


        for framerate in [10,30,60,120,240]:
            self.ui.comboBoxFramerate.addItem(str(framerate))
            self.ui.comboBoxFramerate.setCurrentText("30")
            self.ui.comboBoxFramerate.currentTextChanged.connect(self.update_camera_framerate)

        self.ui.sliderCameraGamma.valueChanged.connect(self.update_camera_gamma)
        self.ui.sliderCameraBrightness.valueChanged.connect(self.update_camera_brightness)
        self.ui.buttonSetFramerate.clicked.connect(self.update_camera_framerate)
        # self.ui.buttonSetFramerate.setVisible(False)

        self.ui.buttonRecord.clicked.connect(self.toggle_recording)


    def get_available_cameras(self):
        self.available_cameras = {c.deviceName() : c.description() for c in QCameraInfo.availableCameras()}
        self.ui.comboBoxCameras.clear()
        for cam in self.available_cameras:
            self.ui.comboBoxCameras.addItem(f"{cam} - {self.available_cameras[cam]}")
        print(self.available_cameras)

    def connect_camera(self, recording = False):
        if hasattr(self, "cam"):
            self.cam.running = False
            time.sleep(0.1)
        device_name = self.ui.comboBoxCameras.currentText().split(" ")[0]

        gamma = self.ui.sliderCameraGamma.value()
        framerate = int(self.ui.comboBoxFramerate.currentText())
        brightness = self.ui.sliderCameraBrightness.value()


        self.cam = Camera(device_name, framerate, gamma, brightness)
        self.cam.signals.signal_frame_changed.connect(self.update_image)
        self.cam.signals.signal_frame_missed.connect(self.frame_missed)
    #
    # if self.ui.checkBoxViewCamera.isChecked():
    #     try:
    #         self.cam.signals.signal_frame_changed.connect(self.update_image)
    #     except:
    #         pass

        self.cam.start()

    def toggle_view_camera(self, view_camera):
        if view_camera:
            # self.cam.signals.signal_frame_changed.connect(self.update_image)
            self.ui.labelVideoDisplay.setVisible(True)
        else:
            # self.cam.signals.signal_frame_changed.disconnect(self.update_image)
            self.ui.labelVideoDisplay.setVisible(False)

    def update_camera_gamma(self, value):
        self.ui.labelCameraGammaValue.setText(str(value/100.0))
        self.cam.gamma = value

    def update_camera_brightness(self, value):
        self.ui.labelCameraBrightnessValue.setText(str(value))
        self.cam.brightness = value
    def update_camera_framerate(self):
        self.connect_camera()
        self.ui.labelFramerate.setText(f"Framerate (currently {self.cam.actual_framerate})")

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        qt_img = self.convert_cv_qt(frame)
        self.ui.labelVideoDisplay.setPixmap(qt_img)

    @QtCore.pyqtSlot(np.ndarray)
    def queue_image(self, frame):
        self.q.put(frame)

    def convert_cv_qt(self, cv_img):
        h, w = cv_img.shape
        ch = 1
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(1200, 1080, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)


    def toggle_recording(self):
        if self.ui.buttonRecord.isChecked():
            #set up a fresh queue
            self.q = Queue()
            self.cam.q = self.q

            #get camera settings to pass writing properties
            h = int(self.cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            framerate = self.cam.actual_framerate

            #retrieve savepath
            savepath = self.ui.lineEditSaveName.text()
            if "." not in savepath:
                savepath += ".avi"
            self.ui.lineEditSaveName.setText(savepath)

            #get video duration from ui
            duration_minutes = self.ui.spinBoxDurationMin.value()
            duration_seconds = self.ui.spinBoxDurationSec.value()
            self.total_duration = duration_minutes * 60 + duration_seconds
            self.total_frames = self.total_duration * framerate

            #set up a timer object
            self.timer = Timer(duration = self.total_duration)
            self.timer.signals.signal_timer_stopped.connect(self.recording_finished)
            self.timer.signals.signal_time_progressed.connect(self.update_capture_progress)

            #set up a videowriter
            self.writer = Writer(q=self.cam.q, savepath = savepath,
                                 framerate=framerate, shape=(w,h))
            self.writer.signal_writing_progress.connect(self.update_writing_progress)



            #start sending frames to queue
            # self.cam.signals.signal_frame_changed.connect(self.queue_image)
            self.cam.recording = True
            #start writing video
            self.writer.start()
            #start timer
            self.timer.start()

            #update ui
            self.ui.buttonRecord.setText("Stop Recording")
            self.ui.buttonRecord.setStyleSheet("color : red")

            self.ui.progressBarCapture.setValue(0)
            self.ui.progressBarWriting.setValue(0)
            self.ui.frameProgress.setVisible(True)
            self.ui.checkBoxViewCamera.setChecked(False)
            sys.stdout.write("\n\n RECORDING BEGAN \n\n")
        else:
            self.recording_finished()

    def recording_finished(self):
        self.cam.recording = False
        self.connect_camera()
        # self.cam.signals.signal_frame_changed.disconnect(self.queue_image)
        self.ui.buttonRecord.setText("Record")
        self.ui.buttonRecord.setStyleSheet("")

        self.ui.progressBarCapture.setValue(100)
        # self.ui.frameProgress.setVisible(False)
        self.ui.buttonRecord.setChecked(False)
        self.ui.checkBoxViewCamera.setChecked(True)
        self.q.put("done")


    @QtCore.pyqtSlot(float)
    def update_capture_progress(self, elapsed):
        self.ui.labelProgressCapture.setText(f"{np.round(elapsed,2)} / {self.total_duration}")
        self.ui.progressBarCapture.setValue(int((elapsed/self.total_duration)*100))
    @QtCore.pyqtSlot(int)
    def update_writing_progress(self, n_frames):
        self.ui.labelProgressWriting.setText(f"Written {n_frames}/{self.total_frames}")
        self.ui.progressBarWriting.setValue(int((n_frames/self.total_frames)*100))
    @QtCore.pyqtSlot(int)
    def frame_missed(self, n_missed_frames):
        sys.stdout.write(f"Missed {n_missed_frames} frames")
    def closeEvent(self, event):
        try:
            self.cam.running = False
        except:
            pass
        print("goodbye")
        event.accept()




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())

