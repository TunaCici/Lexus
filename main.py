"""
Made by Project Lexus Team
Name: main.py
Purpose: Controls each module
and provides logic for the progam.

Author: Tuna Cici
Created: 19/08/2021
"""

import time
import cv2

from modules import logger
from modules import config
from modules import ai
from modules import vibration
from modules import voice_command
from modules import camere

update_rate = 10.0 # update(s) per second
print(f"{1/update_rate * 1000}ms per run")

# entry point
if __name__ == "__main__":
    # logger setup
    main_logger = logger.LexusLogger()
    main_logger.log_info("Starting Project Lexus...")
    main_logger.log_info(f"Current project directory: {config.PROJECT_DIR}")

    # modules setup
    main_ai = ai.Lexus_AI()
    main_vib = vibration.Vibration()
    main_sound = voice_command.VoiceCommander()
    main_camera = camera.Camera()

    # timers for sync and performance
    curr_time = time.perf_counter()
    prev_time = time.perf_counter()
    ai_last_run = time.perf_counter()
    vib_last_run = time.perf_counter()
    sound_last_run = time.perf_counter()
    camera_last_run = time.perf_counter()

    # main loop start
    while True:
        try:
            # update time
            curr_time = time.perf_counter()
            
            # TODO: Design the program flow.
            if (1 / update_rate) <= (curr_time - prev_time):
                """------------- CYCLE START -------------"""
                # TODO: Check if modules are alive
                if not ai.running():
                    main_logger.log_info("AI failed to load.")
                    continue

                img = main_ai.get_image()
                detections = main_ai.get_detections()

                print(detections)

                prev_time = time.perf_counter()

                # TODO: Update the modules.
                main_ai.update(cv2.imread("dog.jpg"))
                """------------- CYCLE END -------------"""

        except KeyboardInterrupt as e:
            main_logger.log_info("Detecting keyboard interrupt.")
            main_logger.log_info("Exitting the program.")
            exit(-13)
        except Exception as e:
            print(e)