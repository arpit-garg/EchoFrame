import asyncio
from asyncio import Task
from fastapi import WebSocket, WebSocketDisconnect
from meeting_connection import MeetingConnection
from utils import utils
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

whisper_flush_interval = 5000

class ConnectionManager:
    connections: dict[str, MeetingConnection]
    flush_audio_task: Task | None

    def __init__(self):
        self.connections = {}
        self.flush_audio_task = None
        logger.debug("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, meeting_id: str | None):
        await websocket.accept()
        self.connections[meeting_id] = MeetingConnection(websocket)
        if self.flush_audio_task is None:
            loop = asyncio.get_running_loop()
            self.flush_audio_task = loop.create_task(self.flush_working_audio_worker())
        logger.info(f"Meeting with id {meeting_id} started. Ongoing meetings: {len(self.connections)}")

    async def process(self, meeting_id: str, chunk: bytes, chunk_timestamp: int):
        logger.debug(f"Processing chunk for meeting {meeting_id}")
        if meeting_id not in self.connections:
            logger.warning(f"No such meeting id {meeting_id}, connection probably closed")
            return
        results = await self.connections[meeting_id].process(chunk, chunk_timestamp)
        await self.send(meeting_id, results)

    async def send(self, meeting_id: str, results: list[utils.TranscriptionResponse] | None):
        if results is not None:
            for result in results:
                try:
                    await self.connections[meeting_id].ws.send_json(result.model_dump())
                    logger.debug(f"Sent transcription result for {meeting_id}: {result.text}")
                except WebSocketDisconnect as e:
                    logger.warning(f"Meeting {meeting_id}: connection closed before sending all results: {e}")
                    self.disconnect(meeting_id)
                except Exception as ex:
                    logger.error(f"Meeting {meeting_id}: exception while sending transcription: {ex}")

    def disconnect(self, meeting_id: str):
        try:
            del self.connections[meeting_id]
            logger.info(f"Disconnected meeting {meeting_id}")
        except KeyError:
            logger.warning(f"Meeting {meeting_id} doesn’t exist anymore")

    async def flush_working_audio_worker(self):
        logger.debug("Starting flush_working_audio_worker")
        while True:
            for meeting_id in self.connections:
                for participant in self.connections[meeting_id].participants:
                    state = self.connections[meeting_id].participants[participant]
                    diff = utils.now() - state.last_received_chunk
                    logger.debug(f"Participant {participant} in {meeting_id} silent for {diff}ms, audio buffer: {len(state.working_audio)} bytes")
                    if diff > whisper_flush_interval and len(state.working_audio) > 0 and not state.is_transcribing:
                        logger.info(f"Forcing transcription in {meeting_id} for {participant}")
                        results = await self.connections[meeting_id].force_transcription(participant)
                        await self.send(meeting_id, results)
            await asyncio.sleep(1)