# This is a Python code for an Ultrasonic Sensor model SC04. 
# 22.08.2021, 21:56
# coded by Neriman Dilara Özcan
 

import RPi.GPIO as GPIO
import time
import config
import threading

# Ultrasonic Sensor Configuration
# First Sensor has TRIGGER = 23 and ECHO = 24
# Second Sensor has TRIGGER = 27 and ECHO = 22

class UltrasonicSensor():
    US_SENSOR_TRIGGER = 0
    US_SENSOR_ECHO = 0

    measurment_max = 0.1

    # last updated distance
    Distance = 0

    # times
    Time = 0
    StartTime = 0
    StopTime = 0
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

    def __init__(self, trigger, echo):
        self.US_SENSOR_TRIGGER = trigger
        self.US_SENSOR_ECHO = echo

    def US_SENSOR_echoInterrupt (self, US_SENSOR_ECHO):
        Time = time.perf_counter()
        if GPIO.input(self.US_SENSOR_ECHO) == 1 :
            self.StartTime = Time
        else :
            self.StopTime = Time

    def US_Sensor_GetDistance(self):
        # Set TRIGGER pin 10 ms to HIGH 
        GPIO.output ( self.US_SENSOR_TRIGGER, True)
        time.sleep(self.Measurment_TRIGGER)
        GPIO.output ( self.US_SENSOR_TRIGGER, False )

        self.StartTime = 0
        self.StopTime = 0

        time.sleep(self.measurment_max)
        
        if self.StartTime < self.StopTime:

            # If measurment is completed 
            Time = self.StopTime - self.StartTime
            self.Distance = Time * self.Measurment_Factor
        else :
            self.Distance = self.Distance_Max_Error

        return self.Distance

    def sensor_thread(self):
        # Configuration of the GPIO ports
        GPIO.setmode ( GPIO.BCM ) # I will specify the port pins to its corresponding port pin numbering
        GPIO.setup ( self.US_SENSOR_TRIGGER, GPIO.OUT )
        GPIO.setup ( self.US_SENSOR_ECHO, GPIO.IN)

        # I am creating an endless loop
    
        GPIO.add_event_detect(self.US_SENSOR_ECHO, GPIO.BOTH, callback = self.US_SENSOR_echoInterrupt)

        try:
            while True:
                self.Distance = round(self.US_Sensor_GetDistance(), 4)

                # sleep 0.2 seconds

                time.sleep(self.Measurment_Pause)
                
                # interrupt  measurment
        except KeyboardInterrupt as e:
            # clean up GPIO port  (yoksa hata verir)
            print("hello")
            print(e)

    # Starting the program with a thread
    def start(self):
        print(f"Trig: {self.US_SENSOR_TRIGGER}, Echo: {self.US_SENSOR_ECHO}")
        my_thread = threading.Thread(
            target=self.sensor_thread
        )
        my_thread.start()
            
    def get_distance(self):
        return self.Distance

if __name__ == "__main__":
    x = UltrasonicSensor(27, 22)
    x.start()

    while True:
        print(f"Distance: {x.get_distance()}")
        time.sleep(0.5)