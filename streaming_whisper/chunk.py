from utils import utils
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class Chunk:
    raw: bytes
    timestamp: int
    duration: float
    size: int
    participant_id: str
    language: str

    def __init__(self, chunk: bytes, chunk_timestamp: int):
        self._extract(chunk)
        self.timestamp = chunk_timestamp
        self.duration = utils.convert_bytes_to_seconds(self.raw)
        self.size = len(self.raw)
        logger.debug(f"Chunk initialized - participant: {self.participant_id}, size: {self.size}")

    def _extract(self, chunk: bytes):
        header = chunk[0:60].decode('utf-8').strip('\x00')
        logger.debug(f"Chunk header: {header}")
        self.raw = chunk[60:]
        header_arr = header.split('|')
        self.participant_id = header_arr[0]
        self.language = utils.get_lang(header_arr[1])