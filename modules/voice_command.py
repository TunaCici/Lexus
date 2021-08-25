from gtts import gTTS
class voice_commander:
    def __init__(self,text1='ağaç',distance=5):
        
        #sistem dosyalarını daha rahat şekilde açmak için
        import os
        
        #Burada kullanacağımız 2 parametre bulunuyor, Dil ve Text
        tts = gTTS(text=str(distance)+'metre sonra önünüze' +text1+'çıkacaktır', lang='tr')
        #Burada oluşturduğumuz ses dosyasını konuma merhaba.mp3 diye kaydediyoruz
        tts.save("merhaba.mp3")

        #şimdi ise bu dosyayı açalım.
        os.system("merhaba.mp3")