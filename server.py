import os
import json
import asyncio
import logging

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bookingbeaver")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
PUBLIC_HOST = os.getenv("PUBLIC_HOST")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

if not PUBLIC_HOST:
    raise RuntimeError("PUBLIC_HOST is not set")

app = FastAPI()

@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    host = PUBLIC_HOST
    stream_url = f"wss://{host}/media-stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connecting you to our AI receptionist.</Say>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>
"""
    return PlainTextResponse(content=twiml, media_type="text/xml")

async def openai_session(websocket_twilio: WebSocket):
    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(uri, extra_headers=headers) as ws_openai:
        session_update = {
            "type": "session.update",
            "session": {
                "model": OPENAI_REALTIME_MODEL,
                "voice": "alloy",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "instructions": (
                    "You are a professional, concise business phone receptionist "
                    "for BookingBeaver."
                ),
            },
        }
        await ws_openai.send(json.dumps(session_update))

        twilio_stream_sid = None

        async def twilio_to_openai():
            nonlocal twilio_stream_sid
            while True:
                msg = await websocket_twilio.receive_text()
                data = json.loads(msg)
                event_type = data.get("event")

                if event_type == "start":
                    twilio_stream_sid = data["start"]["streamSid"]

                elif event_type == "media":
                    payload = data["media"]["payload"]
                    await ws_openai.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": payload
                    }))

                elif event_type == "stop":
                    await ws_openai.send(json.dumps({
                        "type": "input_audio_buffer.commit"
                    }))
                    break

        async def openai_to_twilio():
            nonlocal twilio_stream_sid
            async for raw in ws_openai:
                event = json.loads(raw)
                event_type = event.get("type")

                if event_type == "response.audio.delta":
                    if not twilio_stream_sid:
                        continue
                    audio_chunk = event.get("delta")
                    if not audio_chunk:
                        continue

                    twilio_frame = {
                        "event": "media",
                        "streamSid": twilio_stream_sid,
                        "media": {
                            "payload": audio_chunk
                        }
                    }
                    await websocket_twilio.send_text(json.dumps(twilio_frame))

        t1 = asyncio.create_task(twilio_to_openai())
        t2 = asyncio.create_task(openai_to_twilio())

        await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await openai_session(websocket)
    finally:
        await websocket.close()

@app.get("/")
async def root():
    return {"status": "ok", "service": "BookingBeaver AI Receptionist"}
