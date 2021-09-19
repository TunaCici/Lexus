import cv2 # pip install opencv-python
import os
import glob

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
else:
    # importing from main and inside the package
    import config

import numpy as np

class Camera:
    # images
    ret = None
    frame = None
    photo_no = 0
    frame_list = list()

    # It is trying time that try number to open camera.

    # video object
    videoCaptureObject = None

    def running(self):
        self.is_running = self.videoCaptureObject.isOpened()
        return self.is_running

    # This function opens the camera
    def open_camera(self):
        try:
            self.videoCaptureObject = cv2.VideoCapture(0)

        except cv2.error as error:
            print("[Error]: {}".format(error))

    # This function finds the number of photos inside the photos folder.
    def __init__(self):
        try:
            self.open_camera()
            self.camera_control = self.running()

        except Exception as e:
            print(f"Opening Camera Failed: {e}")
        
        self.photo_no = 0

    # This function returns frame which taken by the camera.
    def get_frame(self):
        return self.frame_list[-1]

    # This function saves the photo which taken by the camera.
    def save(self):
        path = config.PROJECT_DIR + "/photos/"
        name = str(self.photo_no) + '.png' # Photo file name
        if self.ret:
            print(os.path.join(path, name))
            cv2.imwrite(os.path.join(path, name), self.get_frame())

    # This function updates the photo, saves it inside the frame and resizes the frame.  
    def update(self):
        dim = (720, 520)
        try:
            self.ret, self.frame = self.videoCaptureObject.read() # This reads the photo from the camera.
            self.frame = cv2.resize(self.frame, dim)
            self.frame_list.append(self.frame)
        
        except:
            print("Camera cannot read the frame!!!")