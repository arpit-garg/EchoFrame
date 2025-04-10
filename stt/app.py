from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transcribe import Transcriber
import asyncio
import json

app = FastAPI()
transcriber = Transcriber(model_size="tiny", device="cpu")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive audio chunk as bytes
            chunk = await websocket.receive_bytes()
            
            # Handle disconnect signal (e.g., single zero byte)
            if len(chunk) == 1 and chunk == b"\x00":
                transcriber.reset()
                await websocket.close()
                break
            
            # Add chunk to transcriber buffer
            transcriber.add_audio(chunk)
            
            # Transcribe (for simplicity, transcribe every chunk; in practice, add silence detection)
            result = transcriber.transcribe(language="en")
            if result["text"]:
                # Send transcription back to client as JSON
                await websocket.send_json({
                    "text": result["text"],
                    "timestamp": result["timestamp"]
                })
                
            # Optional: Add a small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        transcriber.reset()
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        transcriber.reset()
        await websocket.close()