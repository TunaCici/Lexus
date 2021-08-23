from picamera import PiCamera
import os
import time
import glob
import keyboard # pip install keyboard
import sys

def Camera():
    camera = PiCamera() # This is the camera object.
    path = r"/home/lexus/Desktop"
    pathFiles = r"/home/lexus/Desktop/*.png"
    
    # These paths for testing
    pathWindows = r"C:\Users\sirri\Desktop\Projects\Teknofest\Photos"
    pathFilesWindows = r"C:\Users\sirri\Desktop\Projects\Teknofest\Photos*.png"
    
    print("Press ESC to Quit...")
    
    camera.start_prewiew() # This starts the camera.
    
    i = 0
    while(True):
        camera.capture('C:\Users\sirri\Desktop\Projects\Teknofest\Photos\%s.png' % i) # This captures the photo.
        i = i + 1
        
        time.sleep(0.1)
        
        # This is for deleting garbage photos. The number inside if statement can be changed.
        if i == 200:
            files = glob.glob(pathFiles) # This reads file names and put them inside a list.
            for f in files:
                os.remove(f) # This removes the file which referred with f.
            i = 0 # We have to initialize i value (Photo Counter) to 0.