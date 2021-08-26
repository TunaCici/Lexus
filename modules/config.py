"""
Made by Project Lexus Team
Name: lexus_config.py
Purpose: Includes setting variables for the project.

Author: Tuna Cici
Created: 22/08/2021
"""

import os

# Debug Screen
DSCREEN_RUNNING = True

# Camera
PHOTO_NUMBER = 200
RESIZE_X = 360
RESIZE_Y = 240
CAMERA_RUNNING = True

# Project
PROJECT_DIR = os.getcwd()

# Controller

# Logger
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOG_FILE_DIR = PROJECT_DIR + "/logs/lexuslogfile.txt"

#Ultrasonic_Sensor
THRESHOLD = 2000 #mm
