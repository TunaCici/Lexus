"""
Name: tester.py
Purpose: provide test functions and values
for the controller.

Author: Team Crossbones
Created: 20/08/2021
"""
import time

from utils import random_quote

from modules import logger
from modules import config
#from modules import camera
#from modules import debugNew
from modules import ai
from modules import ultrasonic_sensor
#from PyQt5 import QtCore, QtGui, QtWidgets, QtTest

def test_logger():
    test_logger = logger.LexusLogger()
    test_logger.log_info("Hello from the tester file.")
    test_logger.log_warning("This is a warning. Careful.")
    test_logger.log_error("Something went wrong. Error!")

def test_config():
    print(config.PROJECT_DIR)

"""
def test_debug():
    import sys
    if config.DSCREEN_RUNNING == True:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = debugNew.DebugScreen()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
"""

def test_camera():
    import cv2

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if True: 
            img = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def test_sensor():
    x = ultrasonic_sensor.UltrasonicSensor(27, 22)
    x.start()

    while True:
        print(f"Distance: {x.get_distance()}")
        time.sleep(1/30)

def test_ai():
    import cv2

    temp_ai =  ai.Lexus_AI()
    temp_ai.run_and_display(cv2.imread("dog.jpg"))
    temp_ai.update(cv2.imread("crossing.jpg"))
    temp_ai.update(cv2.imread("no-time-to-die.jpg"))
    temp_ai.update(cv2.imread("running.jpg"))
    temp_ai.update(cv2.imread("street.jpg"))
    temp_ai.update(cv2.imread("drawing.png"))
    temp_ai.update(cv2.imread("john_wick.png"))

def test_random_quote():
    test_logger = logger.LexusLogger()
    r_quote1 = random_quote.get_quote()
    r_quote2 = random_quote.get_quote()

    test_logger.log_info("Testing random_quote utility.")
    test_logger.log_info(r_quote1)
    test_logger.log_info(r_quote2)

if __name__ == "__main__":
    #test_logger()
    #test_config()
    #test_random_quote()
    #test_debug()
    test_camera()
    #test_ai()
    #test_sensor()
