"""
Name: tester.py
Purpose: provide test functions and values
for the controller.

Author: Team Crossbones
Created: 20/08/2021
"""

from modules import logger
from modules import config

def test_logger():
    test_logger = logger.LexusLogger()
    test_logger.log_info("Hello from the tester file.")
    test_logger.log_warning("This is a warning. Careful.")
    test_logger.log_error("Something went wrong. Error!")

def test_config():
    print(config.PROJECT_DIR)

if __name__ == "__main__":
    test_logger()
    #test_config()