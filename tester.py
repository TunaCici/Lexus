"""
Name: tester.py
Purpose: provide test functions and values
for the controller.

Author: Team Crossbones
Created: 20/08/2021
"""
from utils import random_quote

from modules import logger
from modules import config
from modules import camera
from modules import debug
from modules import ai
#from modules import Ultrasonic_Sensor
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest

def test_logger():
    test_logger = logger.LexusLogger()
    test_logger.log_info("Hello from the tester file.")
    test_logger.log_warning("This is a warning. Careful.")
    test_logger.log_error("Something went wrong. Error!")

def test_config():
    print(config.PROJECT_DIR)

def test_debug():
    import sys
    if config.DSCREEN_RUNNING == True:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = debug.DebugScreen()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())

"""
def test_camera():
    test_camera = camera.Camera()
    test_camera.update()
"""

def test_sensor():
    print("")

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
    test_debug()
    #test_camera()
    #test_ai()