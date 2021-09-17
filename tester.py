#Bibliotheken einbinden
import RPi.GPIO as GPIO
import time
# Ultraschall Sensor Konfiguration
US_SENSOR_TRIGGER = 23
US_SENSOR_ECHO = 24
Messung_Max = 1               # in sekunden
Messung_Trigger = 0.00001     # in sekunden
Messung_Pause = 0.2           # in sekunden
Messung_Faktor = (343460 / 2) # Schallgeschwindigkeit durch 2 (hin und zurück) in mm/s
        # Schallgeschwindigkeit gem. https://de.wikipedia.org/wiki/Schallgeschwindigkeit
        # +20°C 343,46m/s
        #   0°C 331,50m/s
        # −20°C 319,09m/s
Abstand_Max = 4000        # Max value in mm
Abstand_Max_Error = Abstand_Max + 1
def US_SENSOR_GetDistance():
    # setze TRIGGER für min 0.01ms
    GPIO.output(US_SENSOR_TRIGGER, True)
    time.sleep(Messung_Trigger)
    GPIO.output(US_SENSOR_TRIGGER, False)
 
    # speichere Startzeit
    StartZeit = time.time()
    MaxZeit = StartZeit + Messung_Max
    # warte aus steigende Flanke von ECHO
    while StartZeit < MaxZeit and GPIO.input(US_SENSOR_ECHO) == 0:
        StartZeit = time.time()
    
    # speichere Stopzeit
    StopZeit = StartZeit
    # warte aus fallende Flanke von ECHO
    while StopZeit < MaxZeit and GPIO.input(US_SENSOR_ECHO) == 1:
        StopZeit = time.time()
    if StopZeit < MaxZeit:
        # berechne Zeitdifferenz zwischen Start und Ankunft im Sekunden
        Zeit = StopZeit - StartZeit
        # berechne Distanz
        Distanz = Zeit * Messung_Faktor
    else:
        # setze Fehlerwert
        Distanz = Abstand_Max_Error
        
    # Distanz als Ganzzahl zurückgeben
    return int(Distanz)
 
if __name__ == '__main__':
    # Ultraschall Sensor Initialisierung GPIO-Pins
    GPIO.setmode(GPIO.BCM)                    # GPIO Modus (BOARD / BCM)
    GPIO.setup(US_SENSOR_TRIGGER, GPIO.OUT)   # Trigger-Pin = Raspberry Pi Output
    GPIO.setup(US_SENSOR_ECHO, GPIO.IN)       # Echo-Pin = raspberry Pi Input
    try:
        while True:
            Abstand = US_SENSOR_GetDistance()
            
            if Abstand >= Abstand_Max:
                # Ausgabe Text
                print ("Kein Objekt gefunden")
            else:
                # Ausgabe Text
                print ("Gemessene Entfernung = %i mm" % Abstand)
            
            time.sleep(Messung_Pause)
 
    # Beim Abbruch durch STRG+C: GPIO Port freigeben
    except KeyboardInterrupt:
        print("Messung vom User gestoppt")
        GPIO.cleanup()