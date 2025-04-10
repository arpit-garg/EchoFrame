import streamlit as st
import whisper

model = whisper.load_model("tiny")



# Title of the app
st.title("My Speech-to-Text App")

st.write("Upload your file")
audio_file = st.file_uploader("choose an audio file", type= ['wav','mp3'])
if audio_file is not None:
    with open('temp_audio.wav',"wb") as f:
        f.write(audio_file.getbuffer())

result = model.transcribe("data hiding paper notebook.wav")
print(result["text"])
# Display the transcribed text (replace this with your actual transcribed text)
transcribed_text = result["text"]
st.write("Transcribed Text:")
st.write(transcribed_text)