"""
Made by Project Lexus Team
Name: random_quote.py
Purpose: Generates random quotes.

Author: Tuna Cici
Created: 22/08/2021
"""

import json
import os

from random import randrange

QUOTE_COUNT = 102

def get_quote() -> str:
    """
    Generates random quote from the /data/quotes.json.
    """
    cwd = os.getcwd()
    f =  open(cwd + "/data/quotes.json", encoding="utf8")

    data = json.load(f)
    quotes : list = data["quotes"]
    r_value = randrange(0, QUOTE_COUNT)
    r_quote = quotes[r_value]
    text = r_quote["quote"] + " -" + r_quote["author"]

    return text
