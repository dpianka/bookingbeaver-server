import os
import json
import base64
import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

import websockets
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = logging.getLogger("bookingbeaver")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------
# Environment / Config
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-realtime-preview-2024-12-17"  # this is what you already used
)

# Google Calendar
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]

calendar_service = None  # initialized on startup

# Simple in-memory lead store (can be replaced with DB later)
LEADS = []

# ---------------------------------------------------------
# Google Calendar helpers
# ---------------------------------------------------------
def _init_calendar_service():
    global calendar_service
    if not GOOGLE_SERVICE_ACCOUNT_JSON or not GOOGLE_CALENDAR_ID:
        logger.warning("Google Calendar not fully configured (missing env vars).")
        calendar_service = None
        return

    try:
        sa_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(sa_info, scopes=GOOGLE_SCOPES)
        calendar_service = build("calendar", "v3", credentials=creds)
        logger.info("Google Calendar service initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Google Calendar service: %s", e)
        calendar_service = None


def find_next_free_slot(duration_minutes: int = 15) -> tuple[datetime, datetime] | None:
    """
    Very simple availability check:
    - Look 2 hours ahead from now
    - Book the first free 15-minute block today
    """
    if not calendar_service:
        logger.warning("find_next_free_slot called but calendar_service is None.")
        return None

    now = datetime.utcnow()
    end_of_window = now + timedelta(hours=8)

    events_result = (
        calendar_service.events()
        .list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now.isoformat() + "Z",
            timeMax=end_of_window.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    # Start searching from 2 hours from now, on 15-min grid
    cursor = now + timedelta(hours=2)
    cursor = cursor.replace(second=0, microsecond=0, minute=(cursor.minute // 15) * 15)

    while cursor < end_of_window:
        slot_end = cursor + timedelta(minutes=duration_minutes)

        conflict = False
        for event in events:
            start_str = event["start"].get("dateTime")
            end_str = event["end"].get("dateTime")
            if not start_str or not end_str:
                continue
            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            # Overlap?
            if not (slot_end <= start or cursor >= end):
                conflict = True
                break

        if not conflict:
            return cursor, slot_end

        cursor += timedelta(minutes=15)

    return None


def create_calendar_event(
    title: str,
    description: str,
    start: datetime,
    end: datetime,
) -> dict | None:
    if not calendar_service:
        logger.warning("create_calendar_event called but calendar_service is None.")
        return None

    event_body = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": TIMEZONE,
        },
    }

    try:
        event = (
            calendar_service.events()
            .insert(calendarId=GOOGLE_CALENDAR_ID, body=event_body)
            .execute()
        )
        logger.info("Created calendar event: %s", event.get("id"))
        return event
    except Exception as e:
        logger.exception("Failed to create calendar event: %s", e)
        return None


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    _init_calendar_service()


@app.get("/")
async def root():
    return {"status": "ok", "service": "BookingBeaver AI Receptionist"}


# ---------------------------------------------------------
# Twilio Voice Webhook
# ---------------------------------------------------------
@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Webhook for incoming Twilio calls."""
    form = await request.form()
    from_number = form.get("From", "unknown")
    logger.info("Incoming call from Twilio: %s", from_number)

    # Figure out our public host to build wss:// URL
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    if not host:
        raise RuntimeError("Cannot determine host for Twilio stream URL")

    stream_url = f"wss://{host}/media-stream"
    logger.info("TwiML connect stream URL: %s", stream_url)

    response = VoiceResponse()
    response.say(
        "Connecting you to our AI receptionist.",
        voice="Polly.Joanna",
    )

    connect = Connect()
    connect.append(Stream(url=stream_url))
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


# ---------------------------------------------------------
# WebSocket bridge: Twilio Media Streams <-> OpenAI Realtime
# ---------------------------------------------------------
async def openai_session(websocket_twilio: WebSocket):
    """
    Bridge audio between Twilio Media Streams and OpenAI Realtime.
    """
    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info("Connecting to OpenAI Realtime WebSocket: %s", uri)

    async with websockets.connect(uri, extra_headers=headers) as websocket_openai:
        logger.info("Connected to OpenAI Realtime")

        # Configure the session so audio formats match Twilio (G.711 µ-law)
        session_update = {
            "type": "session.update",
            "session": {
                "model": OPENAI_REALTIME_MODEL,
                "modalities": ["audio", "text"],
                "voice": "alloy",
                "instructions": (
                    "You are a professional, concise phone receptionist for BookingBeaver. "
                    "You answer calls for small service businesses, greet politely, "
                    "gather the caller's name, phone number, and business name, and "
                    "keep replies short and easy to understand over the phone. "
                    "Never say that you are an AI, and never mention internal tools, APIs, "
                    "tokens, or system internals."
                ),
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
        }

        await websocket_openai.send(json.dumps(session_update))
        logger.info("Sending session.update to OpenAI")

        # Optional: have the assistant start with a greeting immediately
        await websocket_openai.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Start the call by briefly introducing yourself as "
                            "the receptionist for BookingBeaver and ask how you can help."
                        )
                    },
                }
            )
        )
        logger.info("Sent initial greeting prompt to OpenAI")

        stream_sid = None

        async def twilio_to_openai():
            nonlocal stream_sid
            try:
                while True:
                    msg_text = await websocket_twilio.receive_text()
                    msg = json.loads(msg_text)
                    event_type = msg.get("event")
                    logger.info("From Twilio: %s -> %s", event_type, msg)

                    if event_type == "start":
                        stream_sid = msg["start"]["streamSid"]
                        logger.info("Twilio stream started: %s", stream_sid)

                    elif event_type == "media":
                        # Forward audio from Twilio to OpenAI
                        audio_b64 = msg["media"]["payload"]
                        await websocket_openai.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }
                            )
                        )

                    elif event_type == "stop":
                        logger.info("Twilio stream stopped event received")
                        # Commit remaining audio buffer
                        await websocket_openai.send(
                            json.dumps({"type": "input_audio_buffer.commit"})
                        )
                        break
            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
            except Exception as e:
                logger.exception("Error in twilio_to_openai: %s", e)

        async def openai_to_twilio():
            try:
                async for message in websocket_openai:
                    evt = json.loads(message)
                    evt_type = evt.get("type")
                    logger.info("From OpenAI: %s -> %s", evt_type, evt)

                    # Stream audio chunks back to Twilio
                    if evt_type == "response.output_audio.delta":
                        if not stream_sid:
                            # We can't send audio before Twilio 'start'
                            continue
                        audio_b64 = evt["delta"]
                        twilio_msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_b64},
                        }
                        await websocket_twilio.send_text(json.dumps(twilio_msg))

                    # You can expand this branch later to parse text, tools, etc.
            except Exception as e:
                logger.exception("Error in openai_to_twilio: %s", e)

        await asyncio.gather(twilio_to_openai(), openai_to_twilio())

        logger.info("Closing OpenAI session bridge")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio will connect here for Media Streams.
    """
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket connected")
    try:
        await openai_session(websocket)
    finally:
        logger.info("Closing Twilio Media Stream WebSocket")
        await websocket.close()


# ---------------------------------------------------------
# Lead + Calendar debug endpoints
# (These are for you, not Twilio. They won't be called from phone.)
# ---------------------------------------------------------
@app.get("/debug/calendar")
async def debug_calendar():
    """Quick check that Calendar is wired correctly."""
    if not calendar_service:
        return JSONResponse(
            {"ok": False, "error": "Calendar not configured or failed to initialize"},
            status_code=500,
        )

    try:
        now = datetime.utcnow()
        events_result = (
            calendar_service.events()
            .list(
                calendarId=GOOGLE_CALENDAR_ID,
                timeMin=now.isoformat() + "Z",
                maxResults=5,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        return {"ok": True, "sample_events": events}
    except Exception as e:
        logger.exception("Error in /debug/calendar: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/debug/book-test")
async def debug_book_test():
    """Manually create a 15-minute test event to verify bookings work."""
    slot = find_next_free_slot(15)
    if not slot:
        return JSONResponse(
            {"ok": False, "error": "No free slot found in window"},
            status_code=400,
        )

    start, end = slot
    event = create_calendar_event(
        "Test BookingBeaver AI appointment",
        "Created by /debug/book-test",
        start,
        end,
    )
    if not event:
        return JSONResponse(
            {"ok": False, "error": "Failed to create event"},
            status_code=500,
        )

    return {"ok": True, "event": event}


@app.get("/leads")
async def list_leads():
    """Placeholder lead warehouse endpoint."""
    return {"count": len(LEADS), "items": LEADS}
