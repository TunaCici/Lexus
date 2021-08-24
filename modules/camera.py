import cv2 # pip install opencv-python
import os
import time
import glob

import config
import logger

from PyQt5 import QtTest

def cameraPhotoCapturer(temp):
    videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW) # This opens the camera.

    path = config.PROJECT_DIR + "/photos/"# Path for photos. It can be changed.
    pathFiles = config.PROJECT_DIR + "/photos/*.png" # Path for photo files. It can be changed.
    
    if config.DEBUG_RUNNER == True:
        ret, frame = videoCaptureObject.read() # This reads the photo from the camera.    
        tempnum = temp
        name = str(tempnum) + '.png' # Photo file name
        cv2.imwrite(os.path.join(path, name), frame) # This writes the photo file into the Photos folder.
        temp = temp + 1
        QtTest.QTest.qWait(100) # This provides write a photo after 0.1 seconds. It can be changed.

        if temp != 200:
            QtTest.QTest.qWait(100)
            cameraPhotoCapturer(temp)

        if temp == 200:
            files = glob.glob(pathFiles) # This reads file names and put them inside a list.
            for f in files:
                os.remove(f) # This removes the file which referred with f.
            QtTest.QTest.qWait(100)
            logger.LexusLogger()
    
    if config.DEBUG_RUNNER == False:
        files = glob.glob(pathFiles) # This reads file names and put them inside a list.
        for f in files:
            os.remove(f) # This removes the file which referred with f.