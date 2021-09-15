import cv2 # pip install opencv-python
import os
import glob

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
else:
    # importing from main and inside the package
    import config

from PyQt5 import QtTest, QtWidgets, QtCore, QtGui
import numpy as np

class Camera:
    # images
    frame = None
    photo_no = 0
    frame_list = list()

    # It is trying time that try number to open camera.
    try_times = 0

    # video object
    videoCaptureObject = None

    # This function opens the camera
    def open_camera(self):
        try:
            self.videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        except cv2.error as error:
            print("[Error]: {}".format(error))

    # This function finds the number of photos inside the photos folder.
    def __init__(self):
        while(self.try_times != 3):
            self.open_camera()
            self.try_times = self.try_times + 1

        if self.try_times == 3:
            print("There is an issue for camera module. Please check it...")
        
        self.photo_no = 0

    # This function returns frame which taken by the camera.
    def get_frame(self):
        return self.frame_list[-1]

    # This function saves the photo which taken by the camera.
    def save(self):
        path = config.PROJECT_DIR + "/photos/"
        name = str(self.photo_no) + '.png' # Photo file name
        if self.ret:
            cv2.imwrite(os.path.join(path, name), self.get_frame())

    # This function updates the photo, saves it inside the frame and resizes the frame.  
    def update(self):
        self.ret, self.frame = self.videoCaptureObject.read() # This reads the photo from the camera.
        self.frame_list.append(self.frame)