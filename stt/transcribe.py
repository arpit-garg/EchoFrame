from faster_whisper import WhisperModel
import numpy as np
from utils import load_audio, convert_bytes_to_seconds

class Transcriber:
    def __init__(self, model_size="tiny", device="cpu"):
        """Initialize the Whisper model."""
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.buffer = b""  # Audio buffer
        self.last_transcription_time = 0

    def add_audio(self, audio_chunk: bytes):
        """Add audio chunk to buffer."""
        self.buffer += audio_chunk

    def transcribe(self, language="en") -> dict:
        """Transcribe the buffered audio and return results."""
        if not self.buffer:
            return {"text": "", "timestamp": 0}

        audio = load_audio(self.buffer)
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=5,
        )
        text = "".join(segment.text for segment in segments)
        timestamp = self.last_transcription_time
        self.last_transcription_time += int(convert_bytes_to_seconds(self.buffer) * 1000)
        self.buffer = b""  # Clear buffer after transcription
        return {"text": text.strip(), "timestamp": timestamp}

    def reset(self):
        """Clear the buffer."""
        self.buffer = b""