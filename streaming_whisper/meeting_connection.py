from itertools import chain
from typing import List
from faster_whisper.tokenizer import Tokenizer
from starlette.websockets import WebSocket
import logging
from cfg import model
from chunk import Chunk
from state import State
from utils import utils

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

whisper_max_finals_in_initial_prompt = max_finals = 2

class MeetingConnection:
    participants: dict[str, State] = {}
    previous_transcription_tokens: List[int]
    previous_transcription_store: List[List[int]]
    tokenizer: Tokenizer | None
    meeting_language: str | None

    def __init__(self, ws: WebSocket):
        self.participants = {}
        self.ws = ws
        self.previous_transcription_tokens = []
        self.previous_transcription_store = []
        self.meeting_language = None
        self.tokenizer = None
        logger.debug("MeetingConnection initialized")

    async def update_initial_prompt(self, previous_payloads: list[utils.TranscriptionResponse]):
        for payload in previous_payloads:
            if payload.type == 'final' and not any(prompt in payload.text for prompt in utils.black_listed_prompts):
                self.previous_transcription_store.append(self.tokenizer.encode(f' {payload.text.strip()}'))
                if len(self.previous_transcription_tokens) > max_finals:
                    self.previous_transcription_store.pop(0)
                self.previous_transcription_tokens = list(chain.from_iterable(self.previous_transcription_store))
                logger.debug(f"Updated initial prompt with: {payload.text}")

    async def process(self, chunk: bytes, chunk_timestamp: int) -> List[utils.TranscriptionResponse] | None:
        a_chunk = Chunk(chunk, chunk_timestamp)
        logger.debug(f"Processing chunk for participant {a_chunk.participant_id}")

        if not self.meeting_language:
            self.meeting_language = a_chunk.language
            self.tokenizer = Tokenizer(model.hf_tokenizer, multilingual=False, task='transcribe', language=self.meeting_language)
            logger.info(f"Set meeting language to {self.meeting_language}")

        if a_chunk.participant_id not in self.participants:
            logger.debug(f"Creating new state for participant {a_chunk.participant_id}")
            self.participants[a_chunk.participant_id] = State(a_chunk.participant_id, a_chunk.language)

        payloads = await self.participants[a_chunk.participant_id].process(a_chunk, self.previous_transcription_tokens)
        if payloads:
            await self.update_initial_prompt(payloads)
            logger.debug(f"Generated payloads: {[p.text for p in payloads]}")
        return payloads

    async def force_transcription(self, participant_id: str):
        if participant_id in self.participants:
            logger.info(f"Forcing transcription for {participant_id}")
            payloads = await self.participants[participant_id].force_transcription(self.previous_transcription_tokens)
            if payloads:
                await self.update_initial_prompt(payloads)
                logger.debug(f"Forced transcription payloads: {[p.text for p in payloads]}")
            return payloads
        return None