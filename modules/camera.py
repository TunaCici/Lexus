import cv2 # pip install opencv-python
import os
import time
import glob
import tkinter as tk
    
def cameraPhotoCapturer(i):
    videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW) # This opens the camera.

    path = r"C:\Users\sirri\Desktop\Projects\Teknofest\Teknofest\Photos" # Path for photos. It can be changed.
    pathFiles = r"C:\Users\sirri\Desktop\Projects\Teknofest\Teknofest\Photos\*.png" # Path for photo files. It can be changed.

    while(True):
        ret, frame = videoCaptureObject.read() # This reads the photo from the camera.
        tempnum = i
        name = str(tempnum) + '.png' # Photo file name
        cv2.imwrite(os.path.join(path, name), frame) # This writes the photo file into the Photos folder.
        i = i + 1
        time.sleep(0.1) # This provides write a photo after 0.1 seconds. It can be changed.

        if i == 200:
            files = glob.glob(pathFiles) # This reads file names and put them inside a list.
            for f in files:
                    os.remove(f) # This removes the file which referred with f.
            i = 0 # We have to initialize i value (Photo Counter) to 0.