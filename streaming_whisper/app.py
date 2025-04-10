from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
from connection_manager import ConnectionManager
from utils import utils

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

ws_connection_manager = ConnectionManager()
app = FastAPI()

@app.websocket('/ws/{meeting_id}')
async def websocket_endpoint(websocket: WebSocket, meeting_id: str, auth_token: str | None = None):
    logger.debug(f"Accepting WebSocket connection for meeting {meeting_id}")
    await ws_connection_manager.connect(websocket, meeting_id)
    try:
        while True:
            try:
                chunk = await websocket.receive_bytes()
                logger.debug(f"Received chunk of size {len(chunk)} for meeting {meeting_id}")
            except Exception as err:
                logger.warning(f"Expected bytes, received something else, disconnecting {meeting_id}. Error: {err}")
                ws_connection_manager.disconnect(meeting_id)
                break
            if len(chunk) == 1 and ord(b'' + chunk) == 0:
                logger.info(f"Received disconnect message for {meeting_id}")
                ws_connection_manager.disconnect(meeting_id)
                break
            await ws_connection_manager.process(meeting_id, chunk, utils.now())
    except WebSocketDisconnect:
        ws_connection_manager.disconnect(meeting_id)
        logger.info(f"Meeting {meeting_id} has ended")