import websockets
import asyncio
import wave
import logging
import json
import sys

# Configure logging with console output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def send_and_receive_audio():
    meeting_id = "test123"
    participant_id = "client_001"
    language = "en"
    uri = f"ws://localhost:8000/ws/{meeting_id}"

    logger.debug("Starting client execution")

    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to {uri}")

            async def send_audio():
                try:
                    with wave.open(r"C:\Users\garga\Desktop\EchoFrame\harvard.wav", "rb") as wav_file:
                        sample_rate = wav_file.getframerate()
                        bit_depth = wav_file.getsampwidth() * 8
                        channels = wav_file.getnchannels()
                        logger.info(f"Audio specs - Rate: {sample_rate}, Depth: {bit_depth}, Channels: {channels}")
                        
                        if sample_rate != 16000 or bit_depth != 16 or channels != 1:
                            logger.error("Audio format mismatch: must be 16kHz, 16-bit, mono")
                            return

                        header = f"{participant_id}|{language}".ljust(60, '\x00').encode("utf-8")
                        audio_data = wav_file.readframes(wav_file.getnframes())
                        chunk_size = 1024

                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i + chunk_size]
                            full_chunk = header + chunk
                            await websocket.send(full_chunk)
                            logger.debug(f"Sent chunk of size {len(full_chunk)}")
                            await asyncio.sleep(0.01)

                        logger.info("Finished sending audio, waiting 2s before disconnect")
                        await asyncio.sleep(2)
                        await websocket.send(b"\x00")
                        logger.info("Sent disconnect signal")
                except Exception as e:
                    logger.error(f"Send audio error: {e}")

            async def receive_transcription():
                while True:
                    try:
                        message = await websocket.recv()
                        logger.debug(f"Raw message received: {message}")
                        transcription_data = json.loads(message)
                        text = transcription_data.get("text", "")
                        timestamp = transcription_data.get("ts", 0)
                        transcription_type = transcription_data.get("type", "interim")
                        participant = transcription_data.get("participant_id", "unknown")
                        logger.info(f"Received {transcription_type} transcription from {participant} at {timestamp}ms: {text}")
                        print(f"[{transcription_type.upper()}] [{participant}] {text}")
                    except websockets.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error(f"Receive error: {e}")
                        break

            logger.debug("Starting send and receive tasks")
            await asyncio.gather(send_audio(), receive_transcription())

    except Exception as e:
        logger.error(f"Client error: {e}")

if __name__ == "__main__":
    logger.debug("Main execution started")
    asyncio.run(send_and_receive_audio())
    logger.debug("Main execution completed")