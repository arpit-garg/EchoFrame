import asyncio
from asyncio import Task

from fastapi import WebSocket, WebSocketDisconnect


from meeting_connection import MeetingConnection
from utils import utils
import logging

whisper_flush_interval = 5000 # The period in milliseconds to flush the buffer after no new spoken audio is detected
class ConnectionManager:
    connections: dict[str, MeetingConnection] #active meeting connections
    flush_audio_task: Task | None  # backgorund tasks to flush idle audio

    def __init__(self):
        self.connections: dict[str, MeetingConnection] = {}
        self.flush_audio_task = None

    async def connect(self, websocket: WebSocket, meeting_id: str| None):
        
        await websocket.accept()
        self.connections[meeting_id] = MeetingConnection(websocket)
        if self.flush_audio_task is None:
            loop = asyncio.get_running_loop()
            self.flush_audio_task = loop.create_task(self.flush_working_audio_worker())
        
        logging.info(f'Meeting with id {meeting_id} started. Ongoing meetings {len(self.connections)}')

    async def process(self, meeting_id: str, chunk: bytes, chunk_timestamp: int):
        logging.debug(f'Processing chunk for meeting {meeting_id}')
        if meeting_id not in self.connections:
            logging.warning(f'No such meeting id {meeting_id}, the connection was probably closed.')
            return
        results = await self.connections[meeting_id].process(chunk, chunk_timestamp)
        await self.send(meeting_id, results)

    async def send(self, meeting_id: str, results: list[utils.TranscriptionResponse] | None):
        if results is not None:
            for result in results:
                try:
                    await self.connections[meeting_id].ws.send_json(result.model_dump())
                except WebSocketDisconnect as e:
                    logging.warning(f'Meeting {meeting_id}: the connection was closed before sending all results: {e}')
                    self.disconnect(meeting_id)
                except Exception as ex:
                    logging.error(f'Meeting {meeting_id}: exception while sending transcription results {ex}')

    def disconnect(self, meeting_id: str):
        try:
            del self.connections[meeting_id]
        except KeyError:
            logging.warning(f'The meeting {meeting_id} doesn\'t exist anymore.')
        
    async def flush_working_audio_worker(self):
        """
        Will force a transcription for all participants that haven't received any chunks for more than `flush_after_ms`
        but have accumulated some spoken audio without a transcription. This avoids merging un-transcribed "left-overs"
        to the next utterance when the participant resumes speaking.
        """
        while True:
            for meeting_id in self.connections:
                for participant in self.connections[meeting_id].participants:
                    state = self.connections[meeting_id].participants[participant]
                    diff = utils.now() - state.last_received_chunk
                    logging.debug(
                        f'Participant {participant} in meeting {meeting_id} has been silent for {diff} ms and has {len(state.working_audio)} bytes of audio'
                    )
                    if diff > whisper_flush_interval and len(state.working_audio) > 0 and not state.is_transcribing:
                        logging.info(f'Forcing a transcription in meeting {meeting_id} for {participant}')
                        results = await self.connections[meeting_id].force_transcription(participant)
                        await self.send(meeting_id, results)
            await asyncio.sleep(1)
