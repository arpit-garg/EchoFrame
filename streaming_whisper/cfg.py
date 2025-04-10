from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# Just use CPU
device = "cpu"
compute_type = "int8"  # Best for CPU: could also be "int8", "int8_float32", or "float32"

# Choose model name or path
model_name = "tiny.en"  # or path to a local model

# Load Silero VAD (speech detector)
vad_model = load_silero_vad(onnx=False)

# Load Whisper model
model = WhisperModel(
    model_name,
    device=device,
    compute_type=compute_type,
    num_workers=1,
)

# Constant to help convert bytes to time (used in streaming)
one_byte_s = 0.00003125

# Print model info
print("====== WHISPER MODEL INFO ======")
print(f"Model: {model_name}")
print(f"Multilingual: {model.model.is_multilingual}")
