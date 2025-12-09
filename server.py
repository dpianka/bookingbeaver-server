import os
import json
import asyncio
import logging

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
import websockets

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("bookingbeaver")

# ---------------------------------------------------------
# Environment / config
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-realtime-preview-2024-10-01",  # override in Railway if needed
)

PUBLIC_HOST = os.getenv("PUBLIC_HOST")  # e.g. bookingbeaver-server-production.up.railway.app
if not PUBLIC_HOST:
    raise RuntimeError("PUBLIC_HOST is not set (e.g. bookingbeaver-server-production.up.railway.app)")

SYSTEM_MESSAGE = (
    "You are a professional, concise AI phone receptionist for BookingBeaver. "
    "You answer calls for small service businesses. "
    "Greet callers politely, gather their name, phone number, and reason for calling, "
    "and keep replies short and easy to understand over the phone. "
    "Never say that you are an AI, and never mention internal tools, APIs, tokens, or system internals."
)

VOICE = "alloy"

# Which OpenAI events to log verbosely
LOG_EVENT_TYPES = [
    "session.created",
    "session.updated",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.created",
    "response.content.done",
    "response.audio.delta",
    "response.done",
    "rate_limits.updated",
    "error",
]

app = FastAPI()


# ---------------------------------------------------------
# Helpers: OpenAI Realtime session config
# ---------------------------------------------------------
async def send_session_update(openai_ws):
    """
    Configure the Realtime session:
    - g711_ulaw in/out (matches Twilio)
    - server VAD
    - voice + instructions
    """
    session_update = {
        "type": "session.update",
        "session": {
            "model": OPENAI_REALTIME_MODEL,
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 600,
            },
        },
    }
    logger.info("Sending session.update to OpenAI")
    await openai_ws.send(json.dumps(session_update))


async def send_initial_greeting(openai_ws):
    """
    Have the AI greet the caller first.
    """
    item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Greet the caller as the BookingBeaver receptionist. "
                        "Briefly explain that you can help take messages or basic details "
                        "for the business, then ask how you can help."
                    ),
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(item))
    await openai_ws.send(json.dumps({"type": "response.create"}))
    logger.info("Sent initial greeting prompt to OpenAI")


# ---------------------------------------------------------
# 1) Twilio Voice webhook: incoming call
# ---------------------------------------------------------
@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """
    Twilio hits this when a call comes in.
    We return TwiML that:
    - Says a short line with Polly voice (Twilio TTS)
    - Connects a Media Stream to /media-stream
    """
    logger.info("Incoming call from Twilio")

    vr = VoiceResponse()
    vr.say("Connecting you to our BookingBeaver AI receptionist.", voice="Polly.Joanna")

    connect = Connect()
    stream_url = f"wss://{PUBLIC_HOST}/media-stream"
    connect.stream(url=stream_url)
    vr.append(connect)

    logger.info("TwiML connect stream URL: %s", stream_url)
    return PlainTextResponse(str(vr), media_type="application/xml")


# ---------------------------------------------------------
# 2) WebSocket bridge: Twilio <-> OpenAI Realtime
# ---------------------------------------------------------
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket endpoint.
    We:
      - Accept Twilio WS
      - Connect to OpenAI Realtime WS
      - Pump audio Twilio -> OpenAI
      - Pump audio deltas OpenAI -> Twilio
    """
    await websocket.accept(subprotocol="audio.twilio.com")
    logger.info("Twilio Media Stream WebSocket connected")

    openai_ws = None

    try:
        # Connect to OpenAI Realtime
        realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        logger.info("Connecting to OpenAI Realtime WebSocket: %s", realtime_url)

        openai_ws = await websockets.connect(
            realtime_url,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        logger.info("Connected to OpenAI Realtime")

        await send_session_update(openai_ws)
        await send_initial_greeting(openai_ws)

        stream_sid = None

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event in ("start", "connected", "stop", "mark"):
                        logger.info("From Twilio: %s -> %s", event, data)
                    elif event == "media":
                        payload_b64 = data["media"]["payload"]  # base64 g711_ulaw

                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": payload_b64,
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    else:
                        logger.info("From Twilio (other event): %s -> %s", event, data)

                    if event == "start":
                        stream_sid = data["start"]["streamSid"]
                        logger.info("Twilio stream started: %s", stream_sid)

                    if event == "stop":
                        logger.info("Twilio stream stopped event received")
                        break

            except Exception as e:
                logger.exception("Error in receive_from_twilio: %s", e)
                if openai_ws and openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid
            try:
                async for raw in openai_ws:
                    try:
                        event = json.loads(raw)
                    except Exception as decode_err:
                        logger.error("Failed to decode OpenAI message: %s", decode_err)
                        logger.debug("Raw message: %r", raw)
                        continue

                    event_type = event.get("type")

                    if event_type in LOG_EVENT_TYPES:
                        logger.info("From OpenAI: %s -> %s", event_type, event)

                    if event_type == "error" or event.get("error"):
                        logger.error("OpenAI error event: %s", json.dumps(event, indent=2))
                        continue

                    if event_type == "response.audio.delta":
                        if not stream_sid:
                            logger.warning("Got audio.delta but no stream_sid yet")
                            continue

                        delta_b64 = event.get("delta")
                        if not delta_b64:
                            continue

                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": delta_b64},
                        }
                        await websocket.send_json(audio_delta)

            except Exception as e:
                logger.exception("Error in send_to_twilio: %s", e)
                try:
                    await websocket.close()
                except Exception:
                    pass

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

    except Exception as outer_e:
        logger.exception("Fatal error in media_stream handler: %s", outer_e)
    finally:
        if openai_ws and openai_ws.open:
            await openai_ws.close()
        logger.info("Closing Twilio Media Stream WebSocket")
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------
# Health + debug endpoints
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "BookingBeaver AI Receptionist (Realtime)",
        "model": OPENAI_REALTIME_MODEL,
        "host": PUBLIC_HOST,
    }


@app.get("/debug/env")
async def debug_env():
    """
    Quick sanity check endpoint for env config (no secrets).
    """
    return JSONResponse(
        {
            "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
            "OPENAI_REALTIME_MODEL": OPENAI_REALTIME_MODEL,
            "PUBLIC_HOST": PUBLIC_HOST,
            "LOG_LEVEL": LOG_LEVEL,
        }
    )
