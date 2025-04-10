import asyncio
import base64
import time
from typing import List
import logging
from chunk import Chunk
from utils import utils

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

return_audio = False

class State:
    def __init__(self, participant_id: str, lang: str = 'en'):
        self.working_audio_starts_at = 0
        self.participant_id = participant_id
        self.silent_chunks = 0
        self.chunk_count = 0
        self.working_audio = b''
        self.lang = lang
        self.long_silence = False
        self.uuid = utils.Uuid7()
        self.transcription_id = str(self.uuid.get())
        self.last_received_chunk = utils.now()
        self.is_transcribing = False
        self.last_speech_timestamp = 0.0
        logger.debug(f"State initialized for {participant_id}")

    def _extract_transcriptions(self, last_pause: utils.CutMark, ts_result: utils.WhisperResult) -> List[utils.TranscriptionResponse]:
        if ts_result is None:
            return []
        results = []
        final = ''
        interim = ''
        final_starts_at = None
        interim_starts_at = None
        for word in ts_result.words:
            space = ' ' if ' ' not in word.word else ''
            if word.end < last_pause.end:
                final_starts_at = word.start if final_starts_at is None else final_starts_at
                final += word.word + space
                logger.debug(f"Participant {self.participant_id}: final is '{final}'")
            else:
                interim_starts_at = word.start if interim_starts_at is None else interim_starts_at
                interim += word.word + space
                logger.debug(f"Participant {self.participant_id}: interim is '{interim}'")

        if final.strip():
            cut_mark_bytes = self.get_num_bytes_for_slicing(last_pause.end)
            if cut_mark_bytes > 0:
                logger.debug(f"Cut mark set at {cut_mark_bytes} bytes")
                final_start_timestamp = self.working_audio_starts_at + int(final_starts_at * 1000)
                final_audio = None
                final_raw_audio = self.trim_working_audio(cut_mark_bytes)
                if return_audio:
                    final_audio_length = utils.convert_bytes_to_seconds(final_raw_audio)
                    final_audio = utils.get_wav_header([final_raw_audio], final_audio_length) + final_raw_audio
                results.append(self.get_response_payload(final, final_start_timestamp, final_audio, True, probability=last_pause.probability))
            else:
                results.append(self.get_response_payload(final + interim, self.working_audio_starts_at + int(ts_result.words[0].start * 1000)))
                return results
        if interim.strip():
            results.append(self.get_response_payload(interim, self.working_audio_starts_at + int(interim_starts_at * 1000)))
        logger.debug(f"Extracted transcriptions: {[r.text for r in results]}")
        return results

    async def force_transcription(self, previous_tokens) -> List[utils.TranscriptionResponse] | None:
        if self.is_transcribing:
            logger.debug(f"Skipping force transcription for {self.participant_id}, already transcribing")
            return None
        ts_result = await self.do_transcription(self.working_audio, previous_tokens)
        if ts_result and ts_result.text.strip():
            results = [
                self.get_response_payload(
                    ts_result.text,
                    int(ts_result.words[0].start * 1000) + self.working_audio_starts_at,
                    None,
                    True,
                    probability=utils.get_phrase_prob(len(ts_result.words) - 1, ts_result.words)
                )
            ]
            logger.info(f"Forced transcription for {self.participant_id}: {ts_result.text}")
            self.reset()
            return results
        self.reset()
        return None

    async def process(self, chunk: Chunk, previous_tokens: list[int]) -> List[utils.TranscriptionResponse] | None:
        await self.add_to_store(chunk, self.working_audio + chunk.raw)
        if not self.long_silence and not self.is_transcribing:
            ts_result = await self.do_transcription(self.working_audio, previous_tokens)
            last_pause = utils.get_cut_mark_from_segment_probability(ts_result)
            results = self._extract_transcriptions(last_pause, ts_result)
            if results:
                logger.info(f"Processed transcription for {self.participant_id}: {[r.text for r in results]}")
                return results
        logger.debug(f"No transcription results for {self.participant_id}")
        return None

    async def add_to_store(self, chunk: Chunk, tmp_working_audio: bytes = b''):
        now_millis = utils.now()
        self.chunk_count += 1
        if not self.working_audio:
            self.working_audio_starts_at = chunk.timestamp - int(chunk.duration * 1000)
        _, speech_timestamps = utils.is_silent(tmp_working_audio)
        logger.debug(f"Speech timestamps for {self.participant_id}: {speech_timestamps}")
        if speech_timestamps and speech_timestamps[-1]['end'] != self.last_speech_timestamp:
            self.last_speech_timestamp = speech_timestamps[-1]['end']
            self.last_received_chunk = now_millis
            self.working_audio = tmp_working_audio
            self.long_silence = False
            self.silent_chunks = 0
            logger.debug(f"Updated working audio for {self.participant_id}, size: {len(self.working_audio)}")
        else:
            self.silent_chunks += 1
            audio_length_seconds = utils.convert_bytes_to_seconds(tmp_working_audio)
            if speech_timestamps and audio_length_seconds - speech_timestamps[-1]['end'] >= 1:
                self.long_silence = True
                logger.debug(f"Long silence detected for {self.participant_id}")

    def trim_working_audio(self, bytes_to_cut: int) -> bytes:
        logger.debug(f"Trimming audio buffer for {self.participant_id}, current size: {len(self.working_audio)}")
        dropped_chunk = self.working_audio[:bytes_to_cut]
        self.working_audio = self.working_audio[bytes_to_cut:]
        if not self.working_audio:
            self.working_audio_starts_at = 0
        logger.debug(f"Audio buffer after trim: {len(self.working_audio)} bytes")
        return dropped_chunk

    def get_response_payload(self, transcription: str, start_timestamp: int, final_audio: bytes | None = None, final: bool = False, **kwargs):
        prob = kwargs.get('probability', 0.5)
        if not self.transcription_id:
            self.transcription_id = str(self.uuid.get(start_timestamp))
        ts_id = self.transcription_id
        if final:
            self.transcription_id = ''
        payload = utils.TranscriptionResponse(
            id=ts_id,
            participant_id=self.participant_id,
            ts=start_timestamp,
            text=transcription,
            audio=base64.b64encode(final_audio).decode('ASCII') if final_audio else '',
            type='final' if final else 'interim',
            variance=prob,
        )
        logger.debug(f"Created payload for {self.participant_id}: {transcription}")
        return payload

    def reset(self):
        logger.debug(f"Resetting working audio for {self.participant_id}")
        self.working_audio_starts_at = 0
        self.working_audio = b''
        self.last_speech_timestamp = 0.0

    @staticmethod
    def get_num_bytes_for_slicing(cut_mark: float) -> int:
        byte_threshold = utils.convert_seconds_to_bytes(cut_mark)
        sliceable_bytes_multiplier, _ = divmod(byte_threshold, 2048)
        sliceable_bytes = sliceable_bytes_multiplier * 2048
        logger.debug(f"Calculated sliceable bytes: {sliceable_bytes}")
        return sliceable_bytes

    async def do_transcription(self, audio: bytes, previous_tokens: list[int]) -> utils.WhisperResult | None:
        self.is_transcribing = True
        logger.debug(f"Starting transcription for {self.participant_id}, audio size: {len(audio)} bytes")
        try:
            loop = asyncio.get_event_loop()
            ts_result = await loop.run_in_executor(None, utils.transcribe, [audio], self.lang, previous_tokens)
            logger.info(f"Transcription result for {self.participant_id}: {ts_result.text if ts_result else 'None'}")
        except Exception as e:
            logger.error(f"Transcription failed for {self.participant_id}: {e}")
            self.is_transcribing = False
            return None
        self.is_transcribing = False
        return ts_result