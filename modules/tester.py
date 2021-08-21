"""
Name: tester.py
Purpose: provide test functions and values
for the controller.

Author: Team Crossbones
Created: 20/08/2021
"""

import logger

def test_logger():
    test_logger = logger.LexusLogger()
    test_logger.log_info("Hello from the tester file.")
    test_logger.log_warning("This is a warning. Careful.")
    test_logger.log_error("Something went wrong. Error!")

if __name__ == "__main__":
    test_logger()