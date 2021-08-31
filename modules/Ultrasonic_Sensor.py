# This is a Python code for an Ultrasonic Sensor model SC04. 
# 22.08.2021, 21:56
# coded by Neriman Dilara Özcan
 

import RPi.GPIO as GPIO
import time
import config 


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

# GetDistance function : 

def US_SENSOR_GetDistance():
    # set TRIGGER pin at least at 0.01ms
    GPIO.output(Port_Pin_TRIGGER, True) # set to high
    time.sleep(Measure_Trigger) #10 ms defined above
    GPIO.output(Port_Pin_TRIGGER, False) # set to low
 
    # saving Start-time
    Starttime = time.time()
    Maxtime = Starttime + Measurment_Max
    # warte aus steigende Flanke von ECHO
    while Starttime < Maxtime and GPIO.input(Port_Pin_ECHO) == 0:
        Starttime = time.time()
    
    # speichere Stopzeit
    Stopptime= Starttime
    # warte aus fallende Flanke von ECHO
    while Stopptime < Maxtime and GPIO.input(Port_Pin_ECHO) == 1:
        Stopptime = time.time()
    if Stopptime < Maxtime:
        # Calculations for time difference between start and arriving in seconds
        Time = Stopptime - Starttime
        # berechne Distanz
        Distanz = Time * Measurment_Faktor
    else:
        # setze Fehlerwert
        Distanz = Distance_Max_Error
        
    # Distanz als Ganzzahl zurückgeben
    return int(Distanz)
 

 # ---- Main Program Begins : 

if __name__ == '__main__':

    # Ultrasonic Sensor Initializations ~GPIO-Pins
      # Configuration of GPIO Port.
    GPIO.setmode(GPIO.BCM)                  
      # GPIO Modus (BOARD / BCM)
    GPIO.setup(Port_Pin_TRIGGER, GPIO.OUT)   # Trigger-Pin = Raspberry Pi Output
    GPIO.setup(Port_Pin_ECHO, GPIO.IN)       # Echo-Pin = raspberry Pi Input

    try:
        while True:
            Distance = US_SENSOR_GetDistance()
            
            # If the distace, which I will get is bigger than than the max distance defined above then message displayed
            if Distance >= Measurment_Max:
                # Ausgabe Text
                print ("No Object found")
            else:
                # Ausgabe Text
                print ("Mesured Distance = %i mm" % Distance) # measured distance in integer in mm
            
            # 0.2 sec  Pause which I have defined above
            time.sleep(Measurement_Pause)
 
    # Stop the program by :  STRG+C
    except KeyboardInterrupt:
        print("Measurement Stopped !")
        GPIO.cleanup()
