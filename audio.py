import pyttsx3

engine = pyttsx3.init()

def speak(msg):
    engine.say(msg)
    engine.runAndWait()