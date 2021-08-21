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
    yazici = logger.LexusLogger()

    yazici.log_info("Hey bir sey ters gitti.")

    # testing if this commit will be verified
