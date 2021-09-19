import RPi.GPIO as GPIO
import time
import threading
class Vibration:
    is_runing= False
    channel= 0
    def __init__(self, channel):
        self.channel = channel
        GPIO.setwarnings(True)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.channel,GPIO.OUT)
            
    def islem(self,number):
        if number < 0:
            return False

        GPIO.output(self.channel, GPIO.HIGH)
        self.is_runing= True
        time.sleep(number)
        GPIO.output(self.channel, GPIO.LOW)
        self.is_runing= False
        
        return True

    def running(self):
        if self.is_runing== True:
            return True
        else:
            return False

    def vibration(self,number):
        a=threading.Thread(target=self.islem, args=(number,))
        a.start()
        a.join()     

        