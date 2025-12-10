import os
import json
import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, Response

from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

import websockets
from websockets.exceptions import ConnectionClosed

from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("bookingbeaver")

# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")

# Realtime model – override via env if needed
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

PUBLIC_HOST = os.getenv("PUBLIC_HOST")
if not PUBLIC_HOST:
    raise RuntimeError("PUBLIC_HOST must be set (Railway domain, without protocol)")

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BookingBeaver")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")

# Google Calendar config
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "")
TIMEZONE_STR = os.getenv("TIMEZONE", "America/New_York")

# Booking parameters
BOOKING_SLOT_MINUTES = int(os.getenv("BOOKING_SLOT_MINUTES", "15"))
BOOKING_WORK_START_HOUR = int(os.getenv("BOOKING_WORK_START_HOUR", "9"))
BOOKING_WORK_END_HOUR = int(os.getenv("BOOKING_WORK_END_HOUR", "17"))
BOOKING_LOOKAHEAD_DAYS = int(os.getenv("BOOKING_LOOKAHEAD_DAYS", "7"))

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]
LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc

# Global Google Calendar service and in-memory lead inbox
calendar_service = None
LEADS: List[Dict] = []

# ---------------------------------------------------------------------------
# System instructions for the Realtime model
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = f"""
You are a professional, concise phone receptionist for {BUSINESS_NAME}, an AI-powered call answering and appointment booking service for small service businesses.

Your goals:
- Greet callers politely and naturally.
- Quickly understand whether they want:
  - Information about what {BUSINESS_NAME} does, or
  - To book a free consultation call with David, or
  - To leave a message.
- Always collect:
  - Caller name
  - Caller phone number
  - Business name
- If they are interested, try to book a 15-minute consultation in one of David's *free* calendar slots using the tools you have:
  - First, call `get_available_slots` to see which times are open.
  - Propose specific options (e.g., "I have tomorrow at 10:00 AM or 2:30 PM Eastern. Which works better?").
  - Once the caller chooses a time, call `book_appointment` with the chosen slot and caller details.
  - Confirm the final date and time out loud.

Behavior rules:
- Never say that you are an AI.
- Never mention internal tools, APIs, Google Calendar, or any technical details.
- Keep responses short and easy to understand over the phone.
- Stay friendly and confident, but do not oversell. Focus on clarity.
- If tools are unavailable or booking fails, apologize briefly, collect their info, and say that David will follow up to confirm a time.
"""

# ---------------------------------------------------------------------------
# OpenAI Realtime logging
# ---------------------------------------------------------------------------

OPENAI_LOG_EVENTS = {
    "session.created",
    "session.updated",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.created",
    "response.done",
    "response.audio.delta",
    "response.output_item.added",
    "error",
}

# ---------------------------------------------------------------------------
# Google Calendar helpers
# ---------------------------------------------------------------------------


def _init_calendar_service() -> None:
    """Initialize the global Google Calendar service from the service account JSON."""
    global calendar_service

    if not GOOGLE_SERVICE_ACCOUNT_JSON or not GOOGLE_CALENDAR_ID:
        logger.warning(
            "Google Calendar not fully configured. "
            "GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_CALENDAR_ID is missing."
        )
        calendar_service = None
        return

    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=GOOGLE_SCOPES
        )
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        calendar_service = service
        logger.info("Google Calendar service initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Google Calendar service: %s", e)
        calendar_service = None


def get_calendar_service():
    return calendar_service


def _parse_event_time(event_time: Dict) -> datetime:
    """Parse a Google Calendar event start/end into a timezone-aware datetime."""
    if "dateTime" in event_time:
        # Timed event
        return datetime.fromisoformat(event_time["dateTime"])
    # All-day event: treat as local midnight of that date
    d = date.fromisoformat(event_time["date"])
    return datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ)


def get_free_slots_for_date(
    target_date: date,
    slot_minutes: int = BOOKING_SLOT_MINUTES,
    max_slots: int = 5,
) -> List[Dict]:
    """Return free slots for a single day as a list of dicts with start/end ISO strings."""
    svc = get_calendar_service()
    if svc is None:
        logger.warning("get_free_slots_for_date called but calendar_service is None")
        return []

    day_start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BOOKING_WORK_START_HOUR,
        0,
        tzinfo=LOCAL_TZ,
    )
    day_end = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BOOKING_WORK_END_HOUR,
        0,
        tzinfo=LOCAL_TZ,
    )

    # Fetch existing events
    events_result = (
        svc.events()
        .list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    busy: List[Tuple[datetime, datetime]] = []
    for e in events:
        start = _parse_event_time(e["start"])
        end = _parse_event_time(e["end"])
        busy.append((start, end))

    slots: List[Dict] = []
    now = datetime.now(LOCAL_TZ)

    slot_start = day_start
    delta = timedelta(minutes=slot_minutes)

    while slot_start + delta <= day_end and len(slots) < max_slots:
        slot_end = slot_start + delta

        # Don't offer slots in the past (for today)
        if slot_end <= now:
            slot_start = slot_end
            continue

        overlap = False
        for b_start, b_end in busy:
            if b_start < slot_end and b_end > slot_start:
                overlap = True
                break

        if not overlap:
            slots.append(
                {
                    "start": slot_start.isoformat(),
                    "end": slot_end.isoformat(),
                }
            )

        slot_start = slot_end

    logger.info(
        "Computed %d free slots for %s", len(slots), target_date.isoformat()
    )
    return slots


def find_next_free_slot(
    duration_minutes: int = BOOKING_SLOT_MINUTES,
) -> Optional[Tuple[datetime, datetime]]:
    """Find the earliest free slot in the next BOOKING_LOOKAHEAD_DAYS days."""
    today = datetime.now(LOCAL_TZ).date()
    for offset in range(0, BOOKING_LOOKAHEAD_DAYS + 1):
        d = today + timedelta(days=offset)
        slots = get_free_slots_for_date(d, duration_minutes, max_slots=1)
        if slots:
            s = datetime.fromisoformat(slots[0]["start"])
            e = datetime.fromisoformat(slots[0]["end"])
            return s, e
    return None


def create_calendar_event(
    summary: str,
    start_dt: datetime,
    end_dt: datetime,
    description: str = "",
) -> Optional[Dict]:
    """Create an event in Google Calendar, if configured."""
    svc = get_calendar_service()
    if svc is None:
        logger.warning("create_calendar_event called but calendar_service is None")
        return None

    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE_STR},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE_STR},
    }

    try:
        event = (
            svc.events()
            .insert(calendarId=GOOGLE_CALENDAR_ID, body=event_body)
            .execute()
        )
        logger.info("Created calendar event: %s", event.get("id"))
        return event
    except Exception as e:
        logger.exception("Failed to create calendar event: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tool implementations exposed to the Realtime model
# ---------------------------------------------------------------------------


def tool_get_available_slots(max_slots: int = 5) -> Dict:
    """Return a small list of upcoming free slots for the next few days."""
    today = datetime.now(LOCAL_TZ).date()
    all_slots: List[Dict] = []

    for offset in range(0, BOOKING_LOOKAHEAD_DAYS + 1):
        d = today + timedelta(days=offset)
        day_slots = get_free_slots_for_date(
            d, slot_minutes=BOOKING_SLOT_MINUTES, max_slots=max_slots
        )
        for s in day_slots:
            all_slots.append(s)
            if len(all_slots) >= max_slots:
                break
        if len(all_slots) >= max_slots:
            break

    return {
        "ok": True,
        "slots": all_slots,
        "slot_minutes": BOOKING_SLOT_MINUTES,
        "timezone": TIMEZONE_STR,
    }


def tool_book_appointment(
    slot_start: str,
    slot_end: str,
    caller_name: str,
    caller_phone: str,
    business_name: str,
    notes: str = "",
) -> Dict:
    """Create a calendar event for the selected slot and capture the lead."""
    if not slot_start or not slot_end:
        return {"ok": False, "error": "missing_slot_times"}

    try:
        start_dt = datetime.fromisoformat(slot_start)
        end_dt = datetime.fromisoformat(slot_end)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=LOCAL_TZ)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=LOCAL_TZ)
    except Exception as e:
        logger.exception("Invalid slot_start/slot_end format: %s", e)
        return {"ok": False, "error": "invalid_slot_format"}

    lead = {
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "caller_name": caller_name,
        "caller_phone": caller_phone,
        "business_name": business_name,
        "notes": notes,
        "slot_start": start_dt.isoformat(),
        "slot_end": end_dt.isoformat(),
    }
    LEADS.append(lead)
    logger.info("Lead stored: %s", lead)

    summary = f"{BUSINESS_NAME} – 15 min consultation with {caller_name or 'Prospective Client'}"
    desc_parts = [
        f"Caller name: {caller_name}",
        f"Caller phone: {caller_phone}",
        f"Business name: {business_name}",
    ]
    if notes:
        desc_parts.append(f"Notes: {notes}")
    description = "\n".join(desc_parts)

    event = create_calendar_event(summary, start_dt, end_dt, description)

    return {
        "ok": True,
        "lead_saved": True,
        "calendar_saved": event is not None,
        "event_id": event.get("id") if event else None,
        "event_link": event.get("htmlLink") if event else None,
    }


# ---------------------------------------------------------------------------
# OpenAI Realtime session configuration
# ---------------------------------------------------------------------------


async def configure_openai_session(openai_ws) -> None:
    """Send session.update and initial greeting to the Realtime model."""
    tools = [
        {
            "type": "function",
            "name": "get_available_slots",
            "description": "Get a list of upcoming free 15-minute consultation slots on David's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_slots": {
                        "type": "integer",
                        "description": "Maximum number of free slots to return (default 5).",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "book_appointment",
            "description": "Book a consultation in a selected free slot for the caller and store their details as a lead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_start": {
                        "type": "string",
                        "description": "ISO 8601 datetime for the chosen slot start (from get_available_slots).",
                    },
                    "slot_end": {
                        "type": "string",
                        "description": "ISO 8601 datetime for the chosen slot end (from get_available_slots).",
                    },
                    "caller_name": {
                        "type": "string",
                        "description": "Caller full name if provided.",
                    },
                    "caller_phone": {
                        "type": "string",
                        "description": "Caller phone number.",
                    },
                    "business_name": {
                        "type": "string",
                        "description": "Name of the caller's business.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Any additional context from the caller.",
                    },
                },
                "required": ["slot_start", "slot_end"],
            },
        },
    ]

    session_update = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_INSTRUCTIONS.strip(),
            "voice": OPENAI_VOICE,
            "modalities": ["audio", "text"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 600,
                "idle_timeout_ms": None,
                "create_response": True,
                "interrupt_response": True,
            },
            "tools": tools,
            "tool_choice": "auto",
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI")

    # Initial greeting so callers don't get dead air
    initial_greeting = {
        "type": "response.create",
        "response": {
            "instructions": (
                f"You are answering a new inbound phone call for {BUSINESS_NAME}. "
                "Greet the caller, explain briefly what you can help with, and ask how you can help."
            )
        },
    }
    await openai_ws.send(json.dumps(initial_greeting))
    logger.info("Sent initial greeting prompt to OpenAI")


# ---------------------------------------------------------------------------
# FastAPI app + Twilio webhook
# ---------------------------------------------------------------------------

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    _init_calendar_service()
    logger.info("Startup complete")


@app.get("/")
async def root():
    return {"status": "ok", "service": f"{BUSINESS_NAME} AI Receptionist"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Twilio Voice webhook – returns TwiML that connects the media stream."""
    form = await request.form()
    from_number = form.get("From", "unknown")
    logger.info("Incoming call from Twilio: %s", from_number)

    stream_url = f"wss://{PUBLIC_HOST}/media-stream"
    logger.info("TwiML connect stream URL: %s", stream_url)

    vr = VoiceResponse()
    vr.say("Connecting you to our receptionist.", voice="Polly.Joanna")
    connect = Connect()
    connect.append(Stream(url=stream_url))
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


# ---------------------------------------------------------------------------
# Twilio <-> OpenAI Realtime bridge
# ---------------------------------------------------------------------------


async def openai_session_bridge(websocket_twilio: WebSocket):
    """Bridge audio between Twilio Media Streams and OpenAI Realtime."""
    realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info("Connecting to OpenAI Realtime WebSocket: %s", realtime_url)

    async with websockets.connect(realtime_url, extra_headers=headers) as websocket_openai:
        logger.info("Connected to OpenAI Realtime")

        await configure_openai_session(websocket_openai)

        stream_sid: Optional[str] = None

        async def from_twilio_to_openai():
            nonlocal stream_sid
            try:
                while True:
                    msg_text = await websocket_twilio.receive_text()
                    msg = json.loads(msg_text)
                    event = msg.get("event")

                    if event != "media":
                        logger.info("From Twilio: %s -> %s", event, msg)

                    if event == "start":
                        stream_sid = msg.get("start", {}).get("streamSid")
                        logger.info("Twilio stream started: %s", stream_sid)

                    elif event == "media":
                        audio_b64 = msg["media"]["payload"]
                        await websocket_openai.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }
                            )
                        )

                    elif event == "stop":
                        logger.info("Twilio stream stopped event received")
                        # Do NOT send input_audio_buffer.commit here – avoids empty buffer errors
                        break

            except WebSocketDisconnect:
                logger.warning("Twilio websocket disconnected")
            except Exception as e:
                logger.exception("Error in from_twilio_to_openai: %s", e)

        async def from_openai_to_twilio():
            nonlocal stream_sid

            processed_calls = set()

            async def run_tool_and_send(tool_name: str, call_id: str, args_str: str):
                try:
                    args = json.loads(args_str or "{}")
                except Exception:
                    logger.warning(
                        "Could not parse function_call arguments for %s: %r",
                        tool_name,
                        args_str,
                    )
                    args = {}

                if call_id in processed_calls:
                    logger.info("Tool call %s already processed, skipping", call_id)
                    return
                processed_calls.add(call_id)

                logger.info(
                    "Executing function_call: %s(%s) call_id=%s",
                    tool_name,
                    args,
                    call_id,
                )

                try:
                    if tool_name == "get_available_slots":
                        max_slots = int(args.get("max_slots", 5))
                        result = tool_get_available_slots(max_slots=max_slots)
                    elif tool_name == "book_appointment":
                        result = tool_book_appointment(
                            slot_start=args.get("slot_start", ""),
                            slot_end=args.get("slot_end", ""),
                            caller_name=args.get("caller_name", ""),
                            caller_phone=args.get("caller_phone", ""),
                            business_name=args.get("business_name", ""),
                            notes=args.get("notes", "") or "",
                        )
                    else:
                        result = {"ok": False, "error": f"unknown_tool:{tool_name}"}
                except Exception as tool_err:
                    logger.exception("Error executing tool %s: %s", tool_err)
                    result = {"ok": False, "error": str(tool_err)}

                try:
                    await websocket_openai.send(
                        json.dumps(
                            {
                                "type": "input_tool_output",
                                "tool_output": {
                                    "tool_call_id": call_id,
                                    "output": result,
                                },
                            }
                        )
                    )
                    logger.info(
                        "Sent tool result for %s (call_id=%s)", tool_name, call_id
                    )
                except Exception as send_err:
                    logger.exception("Failed to send tool result: %s", send_err)

            try:
                async for raw in websocket_openai:
                    try:
                        evt = json.loads(raw)
                    except Exception as decode_err:
                        logger.error("Failed to decode OpenAI message: %s", decode_err)
                        logger.debug("Raw message: %r", raw)
                        continue

                    evt_type = evt.get("type")

                    if evt_type in OPENAI_LOG_EVENTS:
                        logger.info("From OpenAI: %s -> %s", evt_type, evt)

                    # Error events
                    if evt_type == "error" or evt.get("error"):
                        logger.error("OpenAI error event: %s", json.dumps(evt))
                        continue

                    # Audio back to Twilio
                    if evt_type == "response.audio.delta":
                        if not stream_sid:
                            logger.warning(
                                "Received audio.delta before stream_sid is set"
                            )
                            continue

                        delta_b64 = evt.get("delta")
                        if not delta_b64:
                            continue

                        twilio_msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": delta_b64},
                        }
                        await websocket_twilio.send_text(json.dumps(twilio_msg))
                        continue

                    # Function call started
                    if evt_type == "response.output_item.added":
                        item = evt.get("item", {})
                        if item.get("type") == "function_call":
                            tool_name = item.get("name")
                            call_id = item.get("call_id")
                            args_str = item.get("arguments") or ""
                            logger.info(
                                "Function call started (output_item.added): %s call_id=%s args=%r",
                                tool_name,
                                call_id,
                                args_str,
                            )
                            if args_str.strip():
                                await run_tool_and_send(tool_name, call_id, args_str)
                        continue

                    # Function call completed with arguments
                    if evt_type == "response.done":
                        response_obj = evt.get("response", {})
                        outputs = response_obj.get("output", [])

                        for item in outputs:
                            if item.get("type") != "function_call":
                                continue

                            tool_name = item.get("name")
                            call_id = item.get("call_id")
                            args_str = item.get("arguments") or "{}"

                            logger.info(
                                "Function call completed (response.done): %s call_id=%s args=%r",
                                tool_name,
                                call_id,
                                args_str,
                            )

                            await run_tool_and_send(tool_name, call_id, args_str)
                        continue

            except ConnectionClosed:
                logger.warning("OpenAI websocket closed")
            except Exception as e:
                logger.exception("Error in from_openai_to_twilio: %s", e)

        await asyncio.gather(from_twilio_to_openai(), from_openai_to_twilio())


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams."""
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket connected")
    try:
        await openai_session_bridge(websocket)
    finally:
        await websocket.close()
        logger.info("Twilio Media Stream WebSocket closed")


# ---------------------------------------------------------------------------
# Debug / helper endpoints
# ---------------------------------------------------------------------------


@app.get("/debug/env")
async def debug_env():
    return {
        "PUBLIC_HOST": PUBLIC_HOST,
        "BUSINESS_NAME": BUSINESS_NAME,
        "OPENAI_REALTIME_MODEL": OPENAI_REALTIME_MODEL,
        "GOOGLE_CALENDAR_CONFIGURED": bool(
            GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_CALENDAR_ID
        ),
    }


@app.get("/debug/free-slots")
async def debug_free_slots():
    today = datetime.now(LOCAL_TZ).date()
    slots = get_free_slots_for_date(today, BOOKING_SLOT_MINUTES, max_slots=5)
    return {"today": today.isoformat(), "slots": slots}


@app.post("/debug/book-test")
async def debug_book_test():
    """Create a test booking in the next free slot, to verify calendar wiring."""
    slot = find_next_free_slot()
    if not slot:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "no_free_slots"},
        )

    start_dt, end_dt = slot
    event = create_calendar_event(
        summary=f"{BUSINESS_NAME} – Test Booking",
        start_dt=start_dt,
        end_dt=end_dt,
        description="Test booking created by /debug/book-test endpoint.",
    )
    if not event:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "calendar_insert_failed"},
        )

    return {
        "ok": True,
        "slot_start": start_dt.isoformat(),
        "slot_end": end_dt.isoformat(),
        "event_id": event.get("id"),
        "event_link": event.get("htmlLink"),
    }


@app.get("/leads")
async def list_leads():
    """Return the in-memory list of captured leads."""
    return {"count": len(LEADS), "leads": LEADS}