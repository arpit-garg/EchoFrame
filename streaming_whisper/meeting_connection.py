from itertools import chain
from typing import List

from faster_whisper.tokenizer import Tokenizer

from starlette.websockets import WebSocket

whisper_max_finals_in_initial_prompt = max_finals = 2

import logging 
from cfg import model
from chunk import Chunk
from state import State
from utils import utils



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

    async def update_initial_prompt(self, previous_payloads: list[utils.TranscriptionResponse]):
        for payload in previous_payloads:
            if payload.type == 'final' and not any(prompt in payload.text for prompt in utils.black_listed_prompts):
                self.previous_transcription_store.append(self.tokenizer.encode(f' {payload.text.strip()}'))
                if len(self.previous_transcription_tokens) > max_finals:
                    self.previous_transcription_store.pop(0)
                # flatten the list of lists
                self.previous_transcription_tokens = list(chain.from_iterable(self.previous_transcription_store))

    async def process(self, chunk: bytes, chunk_timestamp: int) -> List[utils.TranscriptionResponse] | None:
        a_chunk = Chunk(chunk, chunk_timestamp)

        # The first chunk sets the meeting language and initializes the Tokenizer
        if not self.meeting_language:
            self.meeting_language = a_chunk.language
            self.tokenizer = Tokenizer(
                model.hf_tokenizer, multilingual=False, task='transcribe', language=self.meeting_language
            )

        if a_chunk.participant_id not in self.participants:
            logging.debug(
                f'The participant {a_chunk.participant_id} is not in the participants list, creating a new state.'
            )
            self.participants[a_chunk.participant_id] = State(a_chunk.participant_id, a_chunk.language)

        payloads = await self.participants[a_chunk.participant_id].process(a_chunk, self.previous_transcription_tokens)
        if payloads:
            await self.update_initial_prompt(payloads)
        return payloads

    async def force_transcription(self, participant_id: str):
        if participant_id in self.participants:
            payloads = await self.participants[participant_id].force_transcription(self.previous_transcription_tokens)
            if payloads:
                await self.update_initial_prompt(payloads)
            return payloads
        return None
