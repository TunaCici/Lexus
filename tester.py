"""
Name: tester.py
Purpose: provide test functions and values
for the controller.

Author: Team Crossbones
Created: 20/08/2021
"""

from modules import logger
from modules import config
from utils import random_quote

def test_logger():
    test_logger = logger.LexusLogger()
    test_logger.log_info("Hello from the tester file.")
    test_logger.log_warning("This is a warning. Careful.")
    test_logger.log_error("Something went wrong. Error!")

def test_config():
    print(config.PROJECT_DIR)

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
    test_random_quote()