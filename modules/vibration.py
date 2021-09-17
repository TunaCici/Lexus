import RPi.GPIO as GPIO
import time
import threading
class Vibration:
    is_runing= False
    def __init__(self,channel):
        if GPIO.input(self.channel):
            self.channel = channel
            GPIO.setWarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(channel,GPIO.out)
        else:
            self.channel = channel
            GPIO.setWarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(channel,GPIO.out)
            

    def islem(self,number):
        if number < 0:
            return False
        GPIO.output(self.channel,GPIO.HIGH)
        self.is_runing= True
        time.sleep(number)
        GPIO.output(self.channel,GPIO.LOW)
        self.is_runing= False
        GPIO.cleanup()
        
    def running(self):
        print("")


    def vibration(self,number):
        a=threading.Thread(target=self.islem, args=(number,))
        a.start()
        a.start()
b = Vibration(17)
b.vibration(10)        

        