from PyQt5 import QtCore, QtWidgets, QtGui
import sys
from Interface.form import Ui_Form
import cv2
from CameraModule.Cam_Light import Camera
import numpy as np

class Main(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.buttonScanForCameras.clicked.connect(self.detect_cameras)
        self.detect_cameras()
        self.ui.spinBoxFramerate.valueChanged.connect(self.update_camera)
        self.ui.sliderCameraBrightness.valueChanged.connect(self.update_camera)
        self.ui.sliderCameraGamma.valueChanged.connect(self.update_camera)

        self.ui.buttonCameraConnectDisconnect.clicked.connect(self.connect_camera)
        self.ui.groupBoxCameraControls.setEnabled(True)

    def detect_cameras(self):
        self.ui.comboBoxConnectedCameras.clear()
        for ii in range(3):
            cap = cv2.VideoCapture(ii)
            if not cap.isOpened():
                pass
            else:
                cam = "Camera "+str(ii +1)
                self.ui.comboBoxConnectedCameras.addItem(cam)


    def update_camera(self):
        try:
            self.cam.running = False
        except:
            print("\nInitializing new camera object")
        camera_number = int(self.ui.comboBoxConnectedCameras.currentText().split(" ")[1]) - 1
        framerate = self.ui.spinBoxFramerate.value()
        brightness = self.ui.sliderCameraBrightness.value()
        gamma = self.ui.sliderCameraGamma.value()
        self.cam = Camera(camera_index=camera_number, framerate=framerate, gamma = gamma)
        self.cam.start()

    def connect_camera(self):
        if not hasattr(self, "cam") or not self.cam.cap.isOpened():
            self.update_camera()
            if self.cam.cap.isOpened():
                self.cam.signal_frame_changed.connect(self.update_image)
                self.ui.buttonCameraConnectDisconnect.setText("Disconnect")
                self.ui.buttonCameraConnectDisconnect.setStyleSheet("color: red")
        else:
            self.cam.running = False
            self.ui.buttonCameraConnectDisconnect.setText("Connect")
            self.ui.buttonCameraConnectDisconnect.setStyleSheet("color: black")

    def update_camera_settings(self):
        self.cam.running = False
        framerate = self.ui.spinBoxFramerate.value()
        brightness = self.ui.sliderCameraBrightness.value()
        self.cam = Camera(camera_index=camera_number, framerate=framerate)


    def setBrightness(self, brightness):
        # self._brightness = brightness
        print(f"Received brighness value {brightness}")
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness))

    def setFramerate(self, framerate):
        print(f"Received framerate value {framerate}")
        self.cap.set(cv2.CAP_PROP_FPS, float(framerate))


    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, frame):
        qt_img = self.convert_cv_qt(frame)
        self.ui.label_2.setPixmap(qt_img)
        # self.ui.label setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb_image.shape
        rgb_image = cv_img
        h, w = cv_img.shape
        ch = 1
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()