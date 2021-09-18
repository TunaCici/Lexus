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
from gtts import gTTS
from time import sleep
import pyglet
from collections import deque 
from queue import Empty, Queue
import threading
import pyglet
import threading
class VoiceCommander:
    # queue
    queue=deque()
    # logger
   #voice_logger = logger.LexusLogger()

    # a flah for cheking if a file is in play
    is_playing = False
    def _init_(self):
        
        tts1 = gTTS(text="Merhaba yol arkadaşım", lang='tr')
        filename1 =config.PROJECT_DIR +"\\data"+"\.mp3"
        tts1.save(filename1)
        music = pyglet.media.load(filename1, streaming=False)
        self.is_playing=True
        music.play()
        self.islem(music.duration)
        os.remove(filename1)
        self.is_playing =False
        
    def _del_(self):
        tts1 = gTTS(text="Bir sonraki yolculuğunda görüşmek üzere yol arkadaşım", lang='tr')
        filename1 =config.PROJECT_DIR +"\\data"+"\.mp3"
        tts1.save(filename1)
        music = pyglet.media.load(filename1, streaming=False)
        music.play()
        self.is_playing=True
        self.islem(music.duration)
        os.remove(filename1)
        self.is_playing=False

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
            filename = vc_file.get("path")+str(i)+".mp3"
            tts.save(filename)
            music = pyglet.media.load(filename, streaming=False)
            self.is_playing= True
            music.play()
            self.islem(music.duration)
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
            self.voice_logger.log_warning("Voice file is empty.")
            return
        if self.queue is not None:      
            vc_file= self.queue[0]
        self.play(vc_file)
        

    def request(self, text : str, priority : str):
        """
        gets some text requests to be played. adds it to the queue accordingly.
        """
        if priority == "high":
            vc_file = {
                "text" : text,
                "path" : config.PROJECT_DIR + "\\data",
                "priority": priority
            }
            # TODO: add this vc_file at the front of the queue
            self.queue.appendleft(vc_file)

        else:
            vc_file = {
                "text" : text,
                "path" : config.PROJECT_DIR + "\\data",
                "priority": priority
            }
            # TODO: add this vc_file at the end of the queue
            self.queue.append(vc_file)