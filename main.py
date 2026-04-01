from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
import random
import string
from typing import Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
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
def get_claude_response(prompt: str, system_prompt: Optional[str] = None):
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_KEY"))
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt or info_prompt(),
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def normalize_code(raw_code: str) -> str:
    return "".join(ch for ch in raw_code.upper() if ch.isalnum())[:8]


def generate_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class CallProfileCreate(BaseModel):
    owner_name: str
    target_name: str
    prompt: str
    code: Optional[str] = None


class CallProfileResponse(BaseModel):
    code: str
    owner_name: str
    target_name: str
    prompt: str


call_profiles: Dict[str, CallProfileResponse] = {}


@app.post("/profiles", response_model=CallProfileResponse)
def create_profile(payload: CallProfileCreate):
    owner_name = payload.owner_name.strip()
    target_name = payload.target_name.strip()
    prompt = payload.prompt.strip()

    if not owner_name or not target_name or not prompt:
        raise HTTPException(status_code=400, detail="owner_name, target_name, and prompt are required.")

    if payload.code:
        code = normalize_code(payload.code)
        if len(code) < 4:
            raise HTTPException(status_code=400, detail="code must be at least 4 characters.")
        if code in call_profiles:
            raise HTTPException(status_code=409, detail="code already exists.")
    else:
        code = generate_code()
        while code in call_profiles:
            code = generate_code()

    profile = CallProfileResponse(
        code=code,
        owner_name=owner_name,
        target_name=target_name,
        prompt=prompt
    )
    call_profiles[code] = profile
    return profile


@app.get("/profiles/{code}", response_model=CallProfileResponse)
def get_profile(code: str):
    normalized = normalize_code(code)
    profile = call_profiles.get(normalized)
    if not profile:
        raise HTTPException(status_code=404, detail="code not found")
    return profile



class WebSocketTranscriber(StreamingClient):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop, system_prompt: str):
        super().__init__(
            StreamingClientOptions(
                api_key=os.environ.get("ASSEMBLYAI_KEY"),
                api_host="streaming.assemblyai.com",
            )
        )
        self.websocket = websocket
        self.loop = loop
        self.system_prompt = system_prompt
        self.previous_response = "-1"
        self.on(StreamingEvents.Begin, self.on_begin)
        self.on(StreamingEvents.Turn, self.on_turn)
        self.on(StreamingEvents.Termination, self.on_terminated)
        self.on(StreamingEvents.Error, self.on_error)

    def on_begin(self, client, event):
        print(f"✅ Session started: {event.id}")

    def on_turn(self, client, event):
        transcript = getattr(event, "transcript", None)
        print((transcript or "") + " " + str(getattr(event, "end_of_turn", False)))
        if transcript and getattr(event, "end_of_turn", False):
            response = get_claude_response(transcript, self.system_prompt)
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


@app.websocket("/stream/{code}")
async def websocket_endpoint(websocket: WebSocket, code: str):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    normalized_code = normalize_code(code)
    profile = call_profiles.get(normalized_code)

    if not profile:
        await websocket.send_text("Invalid code. Ask the owner for the correct connection code.")
        await websocket.close(code=1008)
        return

    audio_queue = queue.Queue()
    stop_signal = object()

    transcriber = WebSocketTranscriber(websocket, loop, profile.prompt)

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