from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
)
import asyncio
import anthropic
import threading
import os
import queue
from dotenv import load_dotenv
from prompt import info_prompt

load_dotenv()



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# get a claude response
def get_claude_response(prompt):
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_KEY"))
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system= info_prompt(),
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text



class WebSocketTranscriber(StreamingClient):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        super().__init__(
            StreamingClientOptions(
                api_key=os.environ.get("ASSEMBLYAI_KEY"),
                api_host="streaming.assemblyai.com",
            )
        )
        self.websocket = websocket
        self.loop = loop
        self.previous_response = "-1"
        self.on(StreamingEvents.Begin, self.on_begin)
        self.on(StreamingEvents.Turn, self.on_turn)
        self.on(StreamingEvents.Termination, self.on_terminated)
        self.on(StreamingEvents.Error, self.on_error)

    def on_begin(self, client, event):
        print(f"✅ Session started: {event.id}")

    def on_turn(self, client, event):
        transcript = getattr(event, "transcript", None)
        print(transcript + " " + str(getattr(event, "end_of_turn", False)))
        if transcript and getattr(event, "end_of_turn", False):
            response = get_claude_response(transcript)
            print("first response: " + response)
            self.previous_response = response
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(f"{response}"),
                self.loop
            )



    def on_terminated(self, client, event):
        print(f"Session ended: {event.audio_duration_seconds}s")

    def on_error(self, client, error):
        print(f"❌ Error: {error}")


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    audio_queue = queue.Queue()
    stop_signal = object()

    transcriber = WebSocketTranscriber(websocket, loop)

    def audio_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is stop_signal:
                break
            yield chunk

    try:
        transcriber.connect(
            StreamingParameters(
                sample_rate=16000,
                encoding=aai.AudioEncoding.pcm_s16le,
                speech_model="universal-streaming-english",
            )
        )

        stream_thread = threading.Thread(
            target=transcriber.stream,
            args=(audio_generator(),),
            daemon=True,
        )
        stream_thread.start()

        while True:
            audio_chunk = await websocket.receive_bytes()
            audio_queue.put(audio_chunk)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        audio_queue.put(stop_signal)
        try:
            transcriber.disconnect(terminate=True)
        except Exception as e:
            print(f"Disconnect error: {e}")