import websockets
import asyncio
import wave
import logging

logging.basicConfig(level=logging.INFO)

async def send_audio():
    meeting_id = "test123"
    uri = f"ws://localhost:8000/ws/{meeting_id}"
    try:
        async with websockets.connect(uri) as websocket:
            logging.info(f"Connected to {uri}")
            with wave.open(r"C:\Users\garga\Desktop\EchoFrame\sample.wav", "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                bit_depth = wav_file.getsampwidth() * 8
                channels = wav_file.getnchannels()
                logging.info(f"Audio specs - Rate: {sample_rate}, Depth: {bit_depth}, Channels: {channels}")
                
                if sample_rate != 16000 or bit_depth != 16 or channels != 1:
                    logging.error("Audio format mismatch")
                    return

                audio_data = wav_file.readframes(wav_file.getnframes())
                chunk_size = 1024

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await websocket.send(chunk)
                    logging.info(f"Sent chunk of size {len(chunk)}")

                await websocket.send(b"\x00")
                logging.info("Sent disconnect signal")
    except Exception as e:
        logging.error(f"Client error: {e}")

asyncio.run(send_audio())