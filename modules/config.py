"""
Made by Project Lexus Team
Name: lexus_config.py
Purpose: Includes setting variables for the project.

Author: Tuna Cici
Created: 22/08/2021
"""

import os

# AI
AI_RUNNING = True


# Debug Screen
DSCREEN_RUNNING = True

# Camera
PHOTO_NUMBER = 200
RESIZE_X = 461
RESIZE_Y = 521
CAMERA_RUNNING = True

# Project
PROJECT_DIR = os.getcwd()

# Controller

# Logger
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOG_FILE_DIR = PROJECT_DIR + "/logs/lexuslogfile.txt"
IS_LOGGER_RUNNING = True
LINE_NUMBER = 0

#Ultrasonic_Sensor
THRESHOLD = 2000 #mm

# Voice Sensor
is_playing = True
