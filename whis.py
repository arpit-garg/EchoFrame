import whisper

model = whisper.load_model("tiny")

result = model.transcribe(r"C:\Users\garga\Desktop\EchoFrame\data hiding paper notebook.wav")

print(result['text'])