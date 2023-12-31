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
from modules import camera
from modules import ultrasonic_sensor

update_rate = 24.0 # update(s) per second
print(f"{1/update_rate * 1000}ms per run")

# entry point
if __name__ == "__main__":
    # logger setup
    main_logger = logger.LexusLogger()
    main_logger.log_info("Starting Project Lexus...")
    main_logger.log_info(f"Current project directory: {config.PROJECT_DIR}")

    # modules setup
    main_ai = ai.Lexus_AI()
    main_vib = vibration.Vibration(17)
    main_sensor_fw = ultrasonic_sensor.UltrasonicSensor(27, 22)
    main_sensor_lf = ultrasonic_sensor.UltrasonicSensor(23, 24)
    main_sound = voice_command.VoiceCommand()
    main_camera = camera.Camera()

    # timers for sync and performance
    curr_time = time.perf_counter()
    prev_time = time.perf_counter()
    ai_last_run = time.perf_counter()
    vib_last_run = time.perf_counter()
    sound_last_run = time.perf_counter()
    camera_last_run = time.perf_counter()
    debug_last_run = time.perf_counter()

    # initialization
    main_sensor_fw.start()
    main_sensor_lf.start()

    # main loop start
    while True:
        try:
            # update time
            curr_time = time.perf_counter()
            
            # TODO: Design the program flow.
            if (1 / update_rate) <= (curr_time - prev_time):
                """------------- CYCLE START -------------"""
                # TODO: Check if modules are alive
                if not main_ai.running():
                    main_logger.log_info("AI failed to load.")
                    continue
                
                # TODO: Update the modules.
                main_sound.update()
                main_camera.update()

                print("start")
                fw_dist = main_sensor_fw.get_distance()
                lf_dist = main_sensor_lf.get_distance()
                time.sleep(0.01)
                print(f"fw_dist: {fw_dist}")
                print(f"lf_dist: {lf_dist}")

                # distance releated
                if fw_dist <= 40.0:
                    main_vib.vibration(0.1)
                if lf_dist <= 20.0:
                    main_vib.vibration(0.4)

                # ai releated
                img = main_camera.get_frame()
                main_ai.update(img)
                detections = main_ai.get_detections()
                if detections:
                    for i in detections:
                        # TODO: Define the rules about detected objects
                        print(i[0])
                        if i[0] == "car":
                            main_sound.request("dikkat, araba görüldü.", "low")
            
                
                prev_time = time.perf_counter()

                """------------- CYCLE END -------------"""

        except KeyboardInterrupt as e:
            cap.release()
            main_logger.log_info("Detecting keyboard interrupt.")
            main_logger.log_info("Exitting the program.")
            exit(-13)
        except Exception as e:
            print(e)
