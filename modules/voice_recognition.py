import speech_recognition as sr
import pyaudio as pa
import locale
import config

class voice_recognition():
    def recorder(self):
        self.r = sr.Recognizer()
        locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            self.data = self.r.record(source, duration=3)
            print("Voice is Recognising...")
            try:
                self.text = self.r.recognize_google(self.data, language = 'tr-TR')
                self.translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")

                self.last_text = self.text.translate(self.translationTable)

                print(self.last_text)
            
            except sr.WaitTimeoutError:
                print("Dinleme zaman asimina ugradi")

            except sr.UnknownValueError:
                print("Ne dedigini anlayamadim")

            except sr.RequestError:
                print("Internete baglanamiyorum")

        return self.last_text

    def update(self):
        try:
            self.sentence = self.recorder()

        except:
            print("There is an error...")

        if self.sentence.lower() == 'kamerayi ac':
            config.CAMERA_RUNNING = True
            print("Correctly Opened...")

        if self.sentence.lower() == 'kamerayi kapat':
            config.CAMERA_RUNNING = False
            print("Correctly Closed...")

        if self.sentence.lower() == 'titresimi ac':
            pass
        
        if self.sentence.lower() == 'titresimi kapat':
            pass
            

obj = voice_recognition()

obj.recorder()