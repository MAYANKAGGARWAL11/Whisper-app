import streamlit as st
import whisper
import tempfile
import os
from textblob import TextBlob
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import wave
st.title("Whisper App")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
text = ""
model = whisper.load_model("base")
st.text("Whisper model loaded")

if st.sidebar.button("Transcribe Audio & Sentimental Analysis"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")

        # Save the uploaded audio file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        transcription = model.transcribe(tmp_file_path)
        st.sidebar.success("Transcription Complete")
        text = transcription["text"]
        st.markdown(transcription["text"])

        os.remove(tmp_file_path)
    else:
        st.sidebar.error("Please upload an audio file")

    st.sidebar.header("Play Original Audio File")
    st.sidebar.audio(audio_file)

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        st.text("Positive sentiment")
    elif polarity < 0:
        st.text("Negative sentiment")
    else:
        st.text("Neutral sentiment")

# Plotting
    fig, ax = plt.subplots()
    ax.plot(range(len(text)), [TextBlob(sent).sentiment.polarity for sent in text], marker='o', linestyle='-')
    ax.set_title('Sentiment Polarity Across Texts')
    ax.set_xlabel('Text Index')
    ax.set_ylabel('Sentiment Polarity')
    ax.set_xticks(range(len(text)))
    ax.set_xticklabels(text, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


    
    