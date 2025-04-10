import secrets
import time
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from pydantic import BaseModel
from silero_vad import get_speech_timestamps, read_audio
from uuid6 import UUID

import cfg as cfg
import logging

# Configuration for Whisper decoding
whisper_beam_size = 1
whisper_min_probability = 0.75

# Set up default logger
log = logging.getLogger(__name__)

# Represents a single word detected by Whisper with its timing and probability
class WhisperWord(BaseModel):
    probability: float
    word: str
    start: float
    end: float

# Represents a segment of transcription output from Whisper
class WhisperSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List

# The transcription response to be sent over the network
class TranscriptionResponse(BaseModel):
    id: str
    participant_id: str
    ts: int
    text: str
    audio: str
    type: str
    variance: float

# A marker where the audio might be split for final transcription
class CutMark(BaseModel):
    start: float = 0.0
    end: float = 0.0
    probability: float = 0.0

# Wraps the result of a Whisper transcription including confidence
class WhisperResult:
    def __init__(self, ts_result):
        self.text = ''.join([segment.text for segment in ts_result])
        self.segments = [WhisperSegment.model_validate(segment._asdict()) for segment in ts_result]
        self.words = [WhisperWord.model_validate(word._asdict()) for segment in ts_result for word in segment.words]
        self.confidence = self.get_confidence()
        self.language = ''

    def __str__(self):
        return (
            f'Text: {self.text}\n'
            + f'Confidence avg: {self.confidence}\n'
            + f'Segments: {self.segments}\n'
            + f'Words: {self.words}'
        )

    def get_confidence(self) -> float:
        if len(self.words) > 0:
            return float(sum(word.probability for word in self.words) / len(self.words))
        return 0.0

# Supported languages (limited for now)
LANGUAGES = {
    "en": "english",
    "hi": "hindi"
}

# Prompts to exclude from context reuse
black_listed_prompts = ['. .']

# Converts audio byte length to seconds
def convert_bytes_to_seconds(byte_str: bytes) -> float:
    return round(len(byte_str) * cfg.one_byte_s, 3)

# Converts time in seconds to byte length
def convert_seconds_to_bytes(cut_mark: float) -> int:
    return int(cut_mark / cfg.one_byte_s)

# Determines if the audio is silent and returns detected speech segments
def is_silent(audio: bytes) -> Tuple[bool, list]:
    chunk_duration = convert_bytes_to_seconds(audio)
    wav_header = get_wav_header([audio], chunk_duration_s=chunk_duration)
    stream = wav_header + audio
    audio_array = read_audio(stream)
    st = get_speech_timestamps(audio_array, model=cfg.vad_model, return_seconds=True)
    log.debug(f'Detected speech timestamps: {st}')
    silent = len(st) == 0
    return silent, st

# Calculates the average probability up to the given index
def get_phrase_prob(last_word_idx: int, words: list[WhisperWord]) -> float:
    word_number = last_word_idx + 1
    return sum([word.probability for word in words[:word_number]]) / word_number

# Finds the biggest pause between words as a candidate for a cut mark
def find_biggest_gap_between_words(word_list: list[WhisperWord]) -> CutMark:
    prev_word = word_list[0]
    biggest_gap_so_far = 0.0
    result = CutMark()
    for i, word in enumerate(word_list):
        if i == 0:
            continue
        diff = word.start - prev_word.end
        probability = get_phrase_prob(i - 1, word_list)
        if diff > biggest_gap_so_far:
            biggest_gap_so_far = diff
            result = CutMark(start=prev_word.end, end=word.start, probability=probability)
            log.debug(f'Biggest gap between words:\n{result}')
        prev_word = word
    return result

# Attempts to find a cut mark based on punctuation, length, or timing
def get_cut_mark_from_segment_probability(ts_result: WhisperResult) -> CutMark:
    check_len = len(ts_result.words) - 1
    phrase = ''
    if len(ts_result.words) > 1:
        if ts_result.words[-1].end >= 10:
            return find_biggest_gap_between_words(ts_result.words)
        for i, word in enumerate(ts_result.words):
            if i == check_len:
                break
            phrase += word.word
            avg_probability = get_phrase_prob(i, ts_result.words)
            if len(phrase) >= 48:
                if (
                    avg_probability >= whisper_min_probability
                    and word.word[-1] in ['.', '!', '?']
                    and word.end < ts_result.words[i + 1].start
                ):
                    log.debug(f'Found split at {word.word} ({word.end} - {ts_result.words[i+1].start})')
                    log.debug(f'Avg probability: {avg_probability}')
                    return CutMark(start=word.end, end=ts_result.words[i + 1].start, probability=avg_probability)
                elif ts_result.words[-1].end >= 15:
                    return find_biggest_gap_between_words(ts_result.words)
    return CutMark()

# Generates a WAV header for a list of audio chunks
def get_wav_header(chunks: List[bytes], chunk_duration_s: float = 0.256, sample_rate: int = 16000) -> bytes:
    duration = len(chunks) * chunk_duration_s
    samples = int(duration * sample_rate)
    bits_per_sample = 16
    channels = 1
    datasize = samples * channels * bits_per_sample // 8
    o = bytes("RIFF", 'ascii')
    o += (datasize + 36).to_bytes(4, 'little')
    o += bytes("WAVE", 'ascii')
    o += bytes("fmt ", 'ascii')
    o += (16).to_bytes(4, 'little')
    o += (1).to_bytes(2, 'little')
    o += channels.to_bytes(2, 'little')
    o += sample_rate.to_bytes(4, 'little')
    o += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')
    o += (channels * bits_per_sample // 8).to_bytes(2, 'little')
    o += bits_per_sample.to_bytes(2, 'little')
    o += bytes("data", 'ascii')
    o += datasize.to_bytes(4, 'little')
    return o

# Converts byte-form audio to float32 numpy array
def load_audio(byte_array: bytes) -> ndarray:
    return np.frombuffer(byte_array, np.int16).flatten().astype(np.float32) / 32768.0

# Returns current UTC time in milliseconds
def now() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

# Runs transcription using Whisper on a list of audio chunks
def transcribe(buffer_list: List[bytes], lang: str = 'en', previous_tokens=None) -> WhisperResult:
    if previous_tokens is None:
        previous_tokens = []
    audio_bytes = b''.join(buffer_list)
    audio = load_audio(audio_bytes)
    iterator, _ = cfg.model.transcribe(
        audio,
        language=lang,
        task='transcribe',
        word_timestamps=True,
        beam_size=whisper_beam_size,
        initial_prompt=previous_tokens,
        condition_on_previous_text=False,
    )
    res = list(iterator)
    ts_obj = WhisperResult(res)
    log.debug(f'Transcription results:\n{ts_obj}\n{res}')
    return ts_obj

# Simplifies language code to match Whisper format
def get_lang(lang: str, short=True) -> str:
    if len(lang) == 2 and short:
        return lang.lower().strip()
    if '-' in lang and short:
        return lang.split('-')[0].strip()
    if not short and '-' in lang:
        split_key = lang.split('-')[0]
        return LANGUAGES.get(split_key, 'english').lower().strip()
    return lang.lower().strip()

# UUIDv7 generator based on current timestamp
class Uuid7:
    def __init__(self):
        self.last_v7_timestamp = None

    def get(self, time_arg_millis: int = None) -> UUID:
        nanoseconds = time.time_ns()
        timestamp_ms = nanoseconds // 10**6

        if time_arg_millis is not None:
            timestamp_ms = time_arg_millis

        if self.last_v7_timestamp is not None and timestamp_ms <= self.last_v7_timestamp:
            timestamp_ms = self.last_v7_timestamp + 1
        self.last_v7_timestamp = timestamp_ms
        uuid_int = (timestamp_ms & 0xFFFFFFFFFFFF) << 80
        uuid_int |= secrets.randbits(76)
        return UUID(int=uuid_int, version=7)

# Extracts JWT from WebSocket headers or fallback parameter
def get_jwt(ws_headers, ws_url_param=None) -> str:
    auth_header = ws_headers.get('authorization', None)
    if auth_header is not None:
        return auth_header.split(' ')[-1]
    return ws_url_param if ws_url_param is not None else ''