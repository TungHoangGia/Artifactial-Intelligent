import os
import time
import playsound
from gtts import gTTS
import speech_recognition as sr

#pip install gTTS
#pip install playsound
#if you can't install playsound, use pip install playsound==1.2.2
#pip install SpeechRecognition
#pip install pyaudio


language = 'en'  # English

def speak(text):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", False)
    os.remove("sound.mp3")

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Me: ", end='')
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio, language="en-US")
            print(text)
            return text
        except:
            print("...")
            return None

while True:
    user_command = get_audio()
    if not user_command:
        continue

    if "hello" in user_command.lower():
        speak("Hi")
        time.sleep(0)

    elif "how are you" in user_command.lower():
        speak("I'm fine, thank you")
        time.sleep(0)
