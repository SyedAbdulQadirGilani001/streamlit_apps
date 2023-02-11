import streamlit as st
from PIL import Image
img = Image.open("st.png")
st.image(img)
video1 = open("Snow Leopards.mp4", "rb") 
st.video(video1)
#st.video(video1, start_time = 25)
audio1 = open("audio.mp3", "rb")
st.audio(audio1)
