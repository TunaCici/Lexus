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
        for files in os.walk(config.PROJECT_DIR + "\\photos\\"):
            for Files in files:
                self.photo_no = self.photo_no + 1

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
        cv2.imwrite(os.path.join(path, name), self.get_frame())
        self.resizer()
        cv2.imwrite(os.path.join(path, name), self.get_frame())
        self.photo_no += 1

    def resizer(self):
        self.before_resizing = cv2.imread(config.PROJECT_DIR + "\\photos\\" + str(self.photo_no) + ".png",1) 
        self.resized = cv2.resize(self.before_resizing,(config.RESIZE_X,config.RESIZE_Y),interpolation = cv2.INTER_AREA)

    # This function updates the photo, saves it inside the frame and resizes the frame.  
    def update(self):
        path = config.PROJECT_DIR + "/photos/"# Path for photos. It can be changed.
        pathFiles = config.PROJECT_DIR + "/photos/*.png" # Path for photo files. It can be changed.
        
        if config.CAMERA_RUNNING == True:
            ret, self.frame = self.videoCaptureObject.read() # This reads the photo from the camera.
            print(self.frame.shape)
            self.save()    
            return