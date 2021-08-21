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
    # TODO: Initialize all modules.

    main_logger.log_info("Everything went smoothly. Have a good day!")
