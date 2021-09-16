import RPi.gpio as GPIO
import time
import threading
class Vibration:
    def __init__(self,channel):
        channel = channel
        GPIO.setWarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(channel,GPIO.out)
    def islem(self,number):
        GPIO.output(self.channel,GPIO.HIGH)
        for i in range(number):
            time.sleep(1)
            pass
        GPIO.output(self.channel,GPIO.LOW)
        GPIO.cleanup()
    def vibration(self,number):
        a=threading.Thread(target=self.islem, args=(number,))
        a.start()
        

        