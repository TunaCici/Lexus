import os
import queue

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger

import os
import time
from gtts import gTTS
from time import sleep
import pyglet
from collections import deque 
from queue import Empty, Queue
import threading
import pyglet
import threading

class VoiceCommand:
    # queue
    queue=deque()
    # logger
    voice_logger = logger.LexusLogger()

    # a flag for cheking if a file is in play
    vc_file = {
                "text" : "",
                "priority": ""
            }
            # 
    is_playing = False

    # last_run time
    last_run = time.perf_counter()

    def __init__(self):
        
        tts1 = gTTS(text="Merhaba yol arkadaşım", lang='tr')
        filename1 ="temp.mp3"
        tts1.save(filename1)
        music = pyglet.media.load(filename1, streaming=False)
        self.is_playing=True
        music.play()
        os.remove(filename1)
        self.is_playing =False
<<<<<<< HEAD
            
=======
    
    def __del__(self):
        print("Program kapandı")

    def islem(second):
        for i in range(second):
            sleep(1)
            pass 
>>>>>>> 1ecc815a33d1588b72f14eccb1708dca81ce585b
    def islem(self,number):
        if number < 0:
            return False
        self.is_playing= True
        sleep(number)
        self.is_playing= False
        
    def play(self, speech : dict):
        """
        plays the selected voice file.
        """
        # how to get data from voice file
        i = 0
        for a in self.queue:
            vc_file= self.queue[i]
            text=vc_file.get("text")
            tts =gTTS(text=text, lang='tr')
            filename = "temp.mp3"
            tts.save(filename)
            music = pyglet.media.load(filename, streaming=False)
            self.is_playing= True
            music.play()
            os.remove(filename)
            i+=1
            self.is_playing= False

        self.is_playing=False
        self.queue.clear() 

    def update(self):
        """
        checks the queue, if there is a voice file, plays it.
        """
        i = 0
        for a in self.queue:
            i+=1
        if i == 0:
            return
        if self.queue is not None:
            vc_file= self.queue[0]
        self.play(vc_file)
        

    def request(self, text : str, priority : str):
        """
        gets some text requests to be played. adds it to the queue accordingly.
        """
        curr_time = time.perf_counter()
        delta = curr_time - self.last_run

        if delta <= 3:
            return False
        
        if priority == "high":
            vc_file = {
                "text" : text,
                "priority": priority
            }
            # TODO: add this vc_file at the front of the queue
            self.queue.appendleft(vc_file)

        else:
            vc_file = {
                "text" : text,
                "priority": priority
            }
            # TODO: add this vc_file at the end of the queue
<<<<<<< HEAD
            self.queue.append(vc_file)

        self.last_run = time.perf_counter()
=======
            self.queue.append(vc_file)
>>>>>>> 1ecc815a33d1588b72f14eccb1708dca81ce585b
