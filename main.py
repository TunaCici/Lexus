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
from modules import Ultrasonic_Sensor
from modules import ai
update_rate = 30.0 # update(s) per second
print(f"{1/update_rate * 1000}ms per run")

# entry point
if __name__ == "__main__":
    main_logger = logger.LexusLogger()
    main_logger.log_info("Starting Project Lexus...")
    main_logger.log_info(f"Current project directory: {config.PROJECT_DIR}")

    main_ai = ai.Lexus_AI()

    curr_time = time.perf_counter()
    prev_time = time.perf_counter()
    elapsed = 0
    # our main loop
    while True:
        try:
            # TODO: Check if modules are alive
            # if they are down. initialize them.
            if True == False:
                print("initialize debugger screen.")
            else:
                None

            # TODO: Design the program flow.
            curr_time = time.perf_counter()
            if (1 / update_rate) <= (curr_time - prev_time):
                elapsed += 1
                if update_rate <= elapsed:
                    # a second has passed
                    print("1 second passed")
                    elapsed = 0
                # ADD THE CODE HERE
                img = main_ai.get_image()
                detections = main_ai.get_detections()

                print(detections)

                prev_time = time.perf_counter()

            # TODO: Update the modules.
            main_ai.update(cv2.imread("dog.jpg"))

        except KeyboardInterrupt as e:
            main_logger.log_info("Detecting keyboard interrupt.")
            main_logger.log_info("Exitting the program.")
            exit(-13)
        except Exception as e:
            print(e)