import os
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
# 3. add a request() function where you will get the text and the priorty add it to the queue
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

class voice_commander:
    # queue
    queue = []

    # logger
    voice_logger = logger.LexusLogger()

    # a flah for cheking if a file is in play
    is_playing = False

    # example voice file
    vc_file = {
        "text:" : "Voice modules is initializing.",
        "path:" : config.PROJECT_DIR + "data/speech/merhaba.mp3",
        "priority": "low"
    }

    def __init__(self,text1='ağaç',distance=5):
        
        #sistem dosyalarını daha rahat şekilde açmak için
        import os
        
        #Burada kullanacağımız 2 parametre bulunuyor, Dil ve Text
        tts = gTTS(text=str(distance)+'metre sonra önünüze' +text1+'çıkacaktır', lang='tr')
        #Burada oluşturduğumuz ses dosyasını konuma merhaba.mp3 diye kaydediyoruz
        tts.save("merhaba.mp3")

        #şimdi ise bu dosyayı açalım.
        os.system("merhaba.mp3")

    def play(self, speech : dict):
        """
        plays the selected voice file.
        """
        if speech is None:
            self.voice_logger.log_warning("Voice file is empty.")
            return
        # how to get data from voice file
        text = speech.get("text")
        path = speech.get("path")
        priority = speech.get("priority")

        # TODO : do your magic

    def update(self):
        """
        checks the queue, if there is a voice file, plays it.
        """
        if self.queue is None:
            return
        vc_file = self.queue.pop(0)
        self.play(vc_file)

    def request(self, text : str, priority : str):
        """
        gets some text requests to be played. adds it to the queue accordingly.
        """
        if priority == "high":
            vc_file = {
                "text:" : text,
                "path:" : config.PROJECT_DIR + "data/speech/text0.mp3",
                "priority": priority
            }
            # TODO: add this vc_file at the front of the queue
            ...

        else:
            vc_file = {
                "text:" : text,
                "path:" : config.PROJECT_DIR + "data/speech/text0.mp3",
                "priority": priority
            }
            # TODO: add this vc_file at the end of the queue
            ...