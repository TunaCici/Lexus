# This is a Python code for an Ultrasonic Sensor model SC04. 
# 22.08.2021, 21:56
# coded by Neriman Dilara Özcan
 

import RPi.GPIO as GPIO
import time
import config 

# Ultrasonic Sensor Configuration
# First Sensor has TRIGGER = 23 and ECHO = 24
# Second Sensor has TRIGGER = 27 and ECHO = 22


US_SENSOR_TRIGGER = 27
US_SENSOR_ECHO = 22

measurment_max = 0.1

# I need a minimal measurment value where I am setting the TRIGGER1 and TRIGGER2 to HIGH

Measurment_TRIGGER = 0.00001 #10 micro seconds
Measurment_Pause = 0.2 # Pause between the measurments (5 measurments per second)

# I need a measurment factor which consists of sound speed
# which I have taken as 20 degrees room temperature +20 degrees = 343,4 m/s
#divided by two because of two measurments ( gidis ve dönüs )
Measurment_Factor = (34360 / 2 )

# I am setting a maximum Measurment value, where the device is giving a error message
Distance_Max = 4000 # 4 meters
Distance_Max_Error = Distance_Max + 1

def US_SENSOR_echoInterrupt ( US_SENSOR_ECHO) :
    global StartTime, StopTime

    Time = time.time()
    if GPIO.input(US_SENSOR_ECHO) == 1 :
        StartTime = Time
    else :
        StopTime = Time

def US_Sensor_GetDistance() :

    # Set TRIGGER pin 10 ms to HIGH 

    global StartTime, StopTime
    GPIO.output ( US_SENSOR_TRIGGER, True)
    time.sleep(Measurment_TRIGGER)
    GPIO.output ( US_SENSOR_TRIGGER, False )

 
    StartTime = 0
    StopTime = 0

    time.sleep(measurment_max)
    
    if StartTime < StopTime :

        # If measurment is completed 
        Time = StopTime - StartTime  
        Distance = Time * Measurment_Factor
    else :
        Distance = Distance_Max_Error

    return Distance


# Starting with the program 

if __name__ == '__main__':

        # Configuration of the GPIO ports

        GPIO.setmode ( GPIO.BCM ) # I will specify the port pins to its corresponding port pin numbering
        GPIO.setup ( US_SENSOR_TRIGGER, GPIO.OUT )
        GPIO.setup ( US_SENSOR_ECHO, GPIO.IN)

# I am creating an endless loop
    
        GPIO.add_event_detect(US_SENSOR_ECHO, GPIO. BOTH, callback = US_SENSOR_echoInterrupt)
        while True:
            Distance = US_Sensor_GetDistance()
            
            if Distance >= Distance_Max:
                print ("No object found")
            
            else  :
                print ( "Measured Distance = %i cm" % Distance)

                # sllep 0.2 seconds

                time.sleep(Measurment_Pause)
                
                # interrupt  measurment

#except KeyboardInterrupt as e:
        #clean up GPIO port  (yoksa hata verir)
       # GPIO.cleanup()
       # print(e)

