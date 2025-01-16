import streamlit as st
import pyttsx3
import speech_recognition as sr
import torch
from transformers import MarianMTModel, MarianTokenizer
from io import BytesIO
import time

# Section 1: Translator Class
class Translator:
    def __init__(self, src_lang="en", tgt_lang="es"):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

# Section 2: Speech-to-Text Function
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    
    # Capture audio from the microphone
    with sr.Microphone() as source:
        st.write("Listening... please speak.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, phrase_time_limit=5)
    
    # Recognize using Sphinx
    try:
        text = recognizer.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        return "Sphinx could not understand the audio."
    except sr.RequestError as e:
        return f"Sphinx error; {e}"

# Section 3: Text-to-Speech Function
def speak_text(text):
    engine = pyttsx3.init('espeak')
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()
    
# Streamlit App UI
st.title("Multifunctional Language App")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Text Translation", "Text-to-Speech", "Speech Recognition"])

# Tab 1: Translation
with tab1:
    st.header("Language Translator")
    src_lang = st.selectbox("Select source language", options=["en", "es", "fr", "de", "it"], key="src")
    tgt_lang = st.selectbox("Select target language", options=["en", "es", "fr", "de", "it"], key="tgt")
    text_to_translate = st.text_area("Enter text to translate", "")

    if st.button("Translate", key="translate"):
        if text_to_translate:
            translator = Translator(src_lang=src_lang, tgt_lang=tgt_lang)
            translated_text = translator.translate(text_to_translate)
            st.write("**Translated Text:**", translated_text)
        else:
            st.warning("Please enter text to translate.")

# Tab 2: Text-to-Speech
with tab2:
    st.header("Text-to-Speech")
    tts_text = st.text_area("Enter text to speak aloud", "")
    if st.button("Speak Text", key="speak"):
        if tts_text:
            st.write("Speaking the text...")
            speak_text(tts_text)
        else:
            st.warning("Please enter text to speak.")

# Tab 3: Speech Recognition
with tab3:
    st.header("Real-Time Speech-to-Text with Sphinx")
    if st.button("Start Recording", key="record"):
        st.write("Recording... please speak.")
        recognized_text = recognize_speech_from_mic()
        st.write("**Recognized Text:**", recognized_text)