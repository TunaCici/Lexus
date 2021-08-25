import cv2 # pip install opencv-python
import os
import time
import glob
import config
import logger

from PyQt5 import QtTest, QtWidgets, QtCore, QtGui

# TEST CASE

# custom_camera = camere.Camera()

# custom_camera.update()
# my_frame = custom_camera.getFrame()


class Camera:
    # images
    frame = None
    resized = None
    photo_no = 0

    # video object
    videoCaptureObject = None

    def __init__(self):
        self.videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        for files in os.walk(config.PROJECT_DIR + "/photos/"):
            for Files in files:
                self.photo_no = self.photo_no + 1

        self.photo_no = 0

    def getFrame(self):
        return self.frame

    def getResized(self):
        return self.resized

    def save(self):
        path = config.PROJECT_DIR + "/photos/"

        name = str(self.photo_no) + '.png' # Photo file name
        cv2.imwrite(os.path.join(path, name), self.resized) # This writes the photo file into the Photos folder.
        self.photo_no += 1
        QtTest.QTest.qWait(100) # This provides write a photo after 0.1 seconds. It can be changed.

    def update(self):
        path = config.PROJECT_DIR + "/photos/"# Path for photos. It can be changed.
        pathFiles = config.PROJECT_DIR + "/photos/*.png" # Path for photo files. It can be changed.
        
        if config.DEBUG_RUNNER == True:
            ret, self.frame = self.videoCaptureObject.read() # This reads the photo from the camera.    
            dim = (256,256)
            self.resized = cv2.resize(self.frame,dim,interpolation = cv2.INTER_AREA)

            return