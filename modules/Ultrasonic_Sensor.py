# This is a Python code for an Ultrasonic Sensor model SC04. 
# 22.08.2021, 21:56
# coded by Neriman Dilara Özcan
 

import os
import time
import RPi.GPIO as GPIO


if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger
class UltrasonicSensor():

    distance = 0

    # Cofiguration of the Ultrasonic Sensor. # 
    # ~Defining some constants

    Port_Pin_TRIGGER = 23 # first port pin, here I have yoused port pin numbered 23 
    Port_Pin_ECHO = 24 #second port pin , here is port pin 24 used

    # Constants for the measurments of the Sensor

    Measurment_Max = 1               # in seconds
    Measure_Trigger = 0.00001     # in seconds ( 10 micro seconds)
    Measurement_Pause = 0.2           # in seconds ( 5 measurments per second )
    # The measurment_Factor is measured by taking the room temperature as referenc. 20 degrees are 343.46m/s so 343460 mm/s 
    Measurment_Faktor = (343460 / 2) # Felling ddivided by 2 (gidis ve gelis) in mm/s
        
    Distance_Max = config.THRESHOLD        # Max value in mm
    Distance_Max_Error = Distance_Max + 1

    def __init__(self):
        # Ultrasonic Sensor Initializations ~GPIO-Pins
        # Configuration of GPIO Port.
        GPIO.setmode(GPIO.BCM)                  
        # GPIO Modus (BOARD / BCM)
        GPIO.setup(self.Port_Pin_TRIGGER, GPIO.OUT)   # Trigger-Pin = Raspberry Pi Output
        GPIO.setup(self.Port_Pin_ECHO, GPIO.IN)       # Echo-Pin = raspberry Pi Input

    # GetDistance function : 

    def US_SENSOR_GetDistance(self):
        # set TRIGGER pin at least at 0.01ms
        GPIO.output(self.Port_Pin_TRIGGER, True) # set to high
        time.sleep(self.Measure_Trigger) #10 ms defined above
        GPIO.output(self.Port_Pin_TRIGGER, False) # set to low
    
        # saving Start-time
        Starttime = time.time()
        Maxtime = Starttime + self.Measurment_Max
        # warte aus steigende Flanke von ECHO
        while Starttime < Maxtime and GPIO.input(self.Port_Pin_ECHO) == 0:
            Starttime = time.time()
        
        # speichere Stopzeit
        Stopptime= Starttime
        # warte aus fallende Flanke von ECHO
        while Stopptime < Maxtime and GPIO.input(self.Port_Pin_ECHO) == 1:
            Stopptime = time.time()
        if Stopptime < Maxtime:
            # Calculations for time difference between start and arriving in seconds
            Time = Stopptime - Starttime
            # berechne Distanz
            Distanz = Time * self.Measurment_Faktor
        else:
            # setze Fehlerwert
            Distanz = self.Distance_Max_Error
            
        # Distanz als Ganzzahl zurückgeben
        return int(Distanz)

    def update(self):
        try:
            self.distance = self.US_SENSOR_GetDistance()
        except Exception:
            pass
    
    def get_distance(self):
        return self.distance