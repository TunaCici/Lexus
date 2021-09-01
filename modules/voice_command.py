import os
import queue
from gtts import gTTS

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger


# TODO:
# 1. class name should be in capitals (ex. VoiceCommander)
# 2. add a queue for voices to be played. (ex. files in the queue will be played one by one)
# 3. add a request() function where you will get the text and the priority add it to the queue
# 3. add a play() function where you will play the selected file/voice
# 4. add an update() function where you check the queue for voices. 
#    (ex. if there is something in the queue. call the play() fuction to play it and then remove it)
# 5. add priority to voice files (ex. if a voice has high priority it will be played first)

# Note: i have written some templates for you.
# Here are example test cases:

# TEST CASE 1
# voice_commander.request("There is a bench in front in front of you, it is close.", "low")
# voice_commander.update()

# in the above code we request a text to be played first.
# then we call the update function to make it play the voice

# TEST CASE 2
# voice_commander.request("There is a cat nearby.", "low")
# voice_commander.request("You are too close to an object on your left side, careful!", "high")
# voice_commander.update()

# in the above code we requst two different text to be played
# you will need to play the high priority text first. even though it came later.

# IMPORTANT NOTE: It is up to you how you implament these features.
#                 I just want to give you some idea how they will play out.
# Also the "voice_file" i wrote is a python dictionary. You can learn about it pretty easily.
# I highly recommend that you learn it, it is very useful.
# A simple implementation of Priority Queue
# using Queue.
from collections import deque 
from queue import Empty, Queue
class VoiceCommander:
    # queue
    queue=deque()
    # logger
    voice_logger = logger.LexusLogger()

    # a flah for cheking if a file is in play
    is_playing = False

    # example voice file
    '''vc_file = {     
        "text:" : "Voice modules is initializing.",
        "path:" : config.PROJECT_DIR + "data/speech/merhaba.mp3",
        "priority": "low"
    }'''
    def __init__(self,text1='ağaç',distance=10):
        
        #sistem dosyalarını daha rahat şekilde açmak için
        import os     
        #Burada kullanacağımız 2 parametre bulunuyor, Dil ve Text
        #Burada oluşturduğumuz ses dosyasını konuma merhaba.mp3 diye kaydediyoruz
        #şimdi ise bu dosyayı açalım.

    def play(self, speech : dict):
        """
        plays the selected voice file.
        """
        if speech is None:
            self.voice_logger.log_warning("Voice file is empty.")
            return
        # how to get data from voice file
        i = 0
        for a in self.queue:
            vc_file= self.queue[i]
            text=vc_file.get("text")
            os.system('espeak -v tr+f2 "{}"'.format(text))
            i+=1
            
        
       
        

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
                "path" : config.PROJECT_DIR + "\\data\\speech",
                "priority": priority
            }
            # TODO: add this vc_file at the front of the queue
            self.queue.appendleft(vc_file)

        else:
            vc_file = {
                "text" : text,
                "path" : config.PROJECT_DIR + "\\data\\speech",
                "priority": priority
            }
            # TODO: add this vc_file at the end of the queue
            self.queue.append(vc_file)
            
    def delete(self):
        self.queue.clear()
        print(self.queue)
