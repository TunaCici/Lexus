"""
Made by Project Lexus Team
Name: main.py
Purpose: Controls each module
and provides logic for the progam.

Author: Tuna Cici
Created: 19/08/2021
"""

from modules import logger
from modules import config

# entry point
if __name__ == "__main__":
    main_logger = logger.LexusLogger()
    main_logger.log_info("Starting Project Lexus...")
    main_logger.log_info(f"Current project directory: {config.PROJECT_DIR}")

    # our main loop
    while True:
        try:
            # TODO: Check if modules are alive
            # if they are down. initialize them.
            if config.DEBUG_RUNNER == False:
                print("initialize debugger screen.")
            else:
                print("")

            # TODO: Design the program flow.

            # TODO: Update the modules.
            
        except KeyboardInterrupt as e:
            main_logger.log_info("Detecting keyboard interrupt.")
            main_logger.log_info("Exitting the program.")
            exit(-13)
        except Exception as e:
            print(e)