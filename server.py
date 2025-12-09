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
    """Twilio webhook for incoming calls."""
    host = PUBLIC_HOST
    stream_url = f"wss://{host}/media-stream"
    logger.info(f"Incoming call -> starting media stream to {stream_url}")

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
    """Bridge audio between Twilio Media Streams and OpenAI Realtime."""
    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info("Connecting to OpenAI Realtime WebSocket...")
    async with websockets.connect(uri, extra_headers=headers) as ws_openai:
        logger.info("Connected to OpenAI Realtime")

        # Enable server-side VAD (turn detection) so the model knows when to reply
        session_update = {
            "type": "session.update",
            "session": {
                "model": OPENAI_REALTIME_MODEL,
                "voice": "alloy",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 500,
                },
                "instructions": (
                    "You are a professional, concise business phone receptionist "
                    "for BookingBeaver, an AI receptionist product. "
                    "Speak clearly, be polite but efficient, and avoid small talk. "
                    "Gather the caller's name, phone number, the reason for their call, "
                    "and any preferred time for service or appointment. "
                    "If the business is closed, politely explain and capture a callback time. "
                    "Never mention that you are an AI model or talk about tokens, APIs, "
                    "or internal systems."
                ),
            },
        }
        await ws_openai.send(json.dumps(session_update))

        twilio_stream_sid = None

        async def twilio_to_openai():
            nonlocal twilio_stream_sid
            try:
                while True:
                    msg = await websocket_twilio.receive_text()
                    data = json.loads(msg)
                    event_type = data.get("event")

                    if event_type == "start":
                        twilio_stream_sid = data["start"]["streamSid"]
                        logger.info(f"Twilio stream started: {twilio_stream_sid}")

                    elif event_type == "media":
                        payload = data["media"]["payload"]
                        # Forward base64-encoded g711_ulaw audio directly
                        await ws_openai.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": payload,
                        }))

                    elif event_type == "stop":
                        logger.info("Twilio stream stopped")
                        # Let OpenAI know there is no more audio coming
                        await ws_openai.send(json.dumps({
                            "type": "input_audio_buffer.commit",
                        }))
                        await ws_openai.send(json.dumps({
                            "type": "response.create",
                        }))
                        break
            except Exception as e:
                logger.warning(f"twilio_to_openai error: {e}")

        async def openai_to_twilio():
            nonlocal twilio_stream_sid
            try:
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
                                "payload": audio_chunk,
                            },
                        }
                        await websocket_twilio.send_text(json.dumps(twilio_frame))

            except Exception as e:
                logger.warning(f"openai_to_twilio error: {e}")

        t1 = asyncio.create_task(twilio_to_openai())
        t2 = asyncio.create_task(openai_to_twilio())
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        logger.info("Closing OpenAI session bridge")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket connected")
    try:
        await openai_session(websocket)
    except Exception as e:
        logger.error(f"media_stream error: {e}")
    finally:
        logger.info("Closing Twilio Media Stream WebSocket")
        await websocket.close()


@app.get("/")
async def root():
    return {"status": "ok", "service": "BookingBeaver AI Receptionist"}
