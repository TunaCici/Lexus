"""
Made by Project Lexus Team
Name: ai.py
Purpose: Talks with the AI and runs it.

Author: Tuna Cici
Created: 25/08/2021
"""

import os

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger

class Lexus_AI():
    bruh = None
    

