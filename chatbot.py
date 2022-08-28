from curses import ERR
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import json
import pickle
import numpy as np
import time
import speech_recognition as sr
import pyttsx3
recognizer = sr.Recognizer()
from gtts import gTTS
from pygame import mixer
mixer.init()

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('data/words.pk1', 'rb'))
classes = pickle.load(open('data/classes.pk1', 'rb'))
model = load_model('data/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res)]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            print("You: ")
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google(audio)
            print(f"----> {text}")
            print("----------------------------------------------")
    except sr.UnknownValueError():
        recognizer = sr.Recognizer()
        continue
    message = text
    ints = predict_class(message)
    print(ints[0]['probability'])
    if float(ints[0]['probability']) >= 0.7:
        res = get_response(ints)
        print(res)
        tts = gTTS(res, lang='en')
        tts.save('data/temp.mp3')
        mixer.music.load('data/temp.mp3')
        mixer.music.play()
        while True:
            if mixer.music.get_busy() == False:
                mixer.music.unload()
                break
        os.remove('data/temp.mp3')
    else:
        print("Sorry, I don't understand.")
        tts = gTTS("Sorry, I don't understand.", lang='en')
        tts.save('data/temp.mp3')
        mixer.music.load('data/temp.mp3')
        mixer.music.play()
        while True:
            if mixer.music.get_busy() == False:
                mixer.music.unload()
                break
        os.remove('data/temp.mp3')