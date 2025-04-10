import numpy as np
from datetime import datetime, timezone

def load_audio(byte_array: bytes) -> np.ndarray:
    """Convert raw audio bytes to a NumPy array for Whisper."""
    return np.frombuffer(byte_array, np.int16).flatten().astype(np.float32) / 32768.0

def now() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def convert_bytes_to_seconds(byte_str: bytes) -> float:
    """Convert audio bytes to seconds (16kHz, 2 bytes/sample)."""
    return len(byte_str) / (16000 * 2)