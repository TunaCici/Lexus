import RPi.GPIO as GPIO
import time
import threading
class Vibration:
    is_runing= False
    def __init__(self,channel):
        self.channel = channel
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(channel,GPIO.OUT)
            

    def islem(self,number,boolean):
        if number < 0:
            return False
        if boolean == 0:    
            GPIO.output(self.channel,GPIO.HIGH)
            self.is_runing= True
            time.sleep(number)
            GPIO.output(self.channel,GPIO.LOW)
            self.is_runing= False
            GPIO.cleanup()
        else:
            print("channel is used")

    def running(self):
        if self.is_runing== True:
            return True
        else:
            return False


    def vibration(self,number):
        a=threading.Thread(target=self.islem, args=(number,self.running()))
        a.start()  
a = Vibration(17)
a.vibration(10)        

        