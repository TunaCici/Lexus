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
    resized = None
    photo_no = 0

    # video object
    videoCaptureObject = None

    # This function opens the camera and finds the number of photos inside the photos folder.
    def __init__(self):
        self.videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.photo_no = 0

    # This function returns frame which taken by the camera.
    def get_frame(self):
        return self.frame

    # This function returns the resized frame.
    def get_resized(self):
        return self.resized

    # This function saves the photo which taken by the camera.
    def save(self):
        path = config.PROJECT_DIR + "/photos/"
        name = str(self.photo_no) + '.png' # Photo file name
        if self.ret:
            cv2.imwrite(os.path.join(path, name), self.get_frame())
            self.resizer()
            cv2.imwrite(os.path.join(path, name), self.get_resized())

    def resizer(self):
        self.before_resizing = cv2.imread(config.PROJECT_DIR + "\\photos\\" + str(self.photo_no) + ".png",1) 
        self.resized = cv2.resize(self.before_resizing,(config.RESIZE_X,config.RESIZE_Y))

    # This function updates the photo, saves it inside the frame and resizes the frame.  
    def update(self):
        path = config.PROJECT_DIR + "/photos/"# Path for photos. It can be changed.
        pathFiles = config.PROJECT_DIR + "/photos/*.png" # Path for photo files. It can be changed.
        self.ret, self.frame = self.videoCaptureObject.read() # This reads the photo from the camera.
        
        if config.CAMERA_RUNNING == True:
            self.save()
            return