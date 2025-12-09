import os
import json
import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import JSONResponse, Response

from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

import websockets
from websockets.exceptions import ConnectionClosed

from google.oauth2 import service_account
from googleapiclient.discovery import build

# =========================================================
# Logging
# =========================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("bookingbeaver")

# =========================================================
# Environment / Config
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")

# Use env if provided; otherwise fallback to a reasonable default name.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

PUBLIC_HOST = os.getenv("PUBLIC_HOST")  # e.g. bookingbeaver-server-production.up.railway.app
if not PUBLIC_HOST:
    raise RuntimeError("PUBLIC_HOST must be set (Railway domain, without protocol)")

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BookingBeaver")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")

# Google Calendar config
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "")
TIMEZONE_STR = os.getenv("TIMEZONE", "America/New_York")

# Booking window config
BOOKING_SLOT_MINUTES = int(os.getenv("BOOKING_SLOT_MINUTES", "15"))
BOOKING_WORK_START_HOUR = int(os.getenv("BOOKING_WORK_START_HOUR", "9"))    # 09:00
BOOKING_WORK_END_HOUR = int(os.getenv("BOOKING_WORK_END_HOUR", "17"))      # 17:00
BOOKING_LOOKAHEAD_DAYS = int(os.getenv("BOOKING_LOOKAHEAD_DAYS", "7"))

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Local timezone
LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc

calendar_service = None  # initialized on startup

# In-memory lead store
LEADS: List[Dict] = []

# =========================================================
# System instructions for the receptionist
# =========================================================

SYSTEM_INSTRUCTIONS = f"""
You are a professional, concise phone receptionist for {BUSINESS_NAME}.

Your main goal:
- Talk with the caller and book a specific consultation time during the call.
- Never just say “we will be in touch later” if you can instead schedule an exact time.

Tools you can use:
- `get_available_slots(max_slots)`:
  - Returns a JSON object with:
    • `slots`: a list of free 15-minute slots {{ "start": ISO8601, "end": ISO8601 }}
    • `slot_minutes`: the slot length in minutes
    • `timezone`: the calendar timezone
  - Use this whenever you need to see what times are available in the calendar.
- `book_appointment(slot_start, slot_end, caller_name, caller_phone, business_name, notes)`:
  - Tries to create a calendar event for that time range.
  - Returns `{{ "ok": true, "event": {{…}} }}` on success, or `{{ "ok": false, "error": "..." }}` on failure.
  - Also records the lead details (name, phone, business) in a lead inbox.

How to handle calls:
1) Greet the caller briefly as the receptionist for {BUSINESS_NAME} and ask how you can help.
2) If they are interested in BookingBeaver or want to talk to a specialist:
   - In one or two short sentences, explain that BookingBeaver is an AI-powered answering and booking assistant that:
     • answers calls 24/7,
     • answers common questions,
     • captures caller details,
     • and integrates with their calendar so appointments are scheduled automatically.
   - Your primary goal is to schedule a short consultation during this call.

3) Lead capture:
   - Ask for:
     • Full name
     • Best callback phone number
     • Business name
   - Confirm these back clearly in one short sentence.
   - If they refuse to give some info, continue with what you have.

4) Scheduling behavior (THIS IS IMPORTANT):
   - After you have their basic info, call `get_available_slots` to see the next free times.
   - Offer 1–3 specific options in simple language, for example:
     “I can do today at 3:00 PM, or tomorrow at 10:00 AM or 2:30 PM Eastern. Which works best for you?”
   - If they suggest a time themselves, choose the closest matching free slot from `get_available_slots`.
   - Once they choose a slot, call `book_appointment` with:
     • the chosen `slot_start` and `slot_end` from the free slots,
     • the caller’s name, phone, and business,
     • a short note like “BookingBeaver consultation call”.
   - If `book_appointment` fails (for example, conflict or error), apologize briefly and call `get_available_slots` again to propose a different time.

5) What to say after booking:
   - After `book_appointment` returns `ok: true`, confirm the booking out loud in one clear sentence:
     - Include day of week, date, time, and “Eastern time”.
     - Confirm their name, business, and phone number.
   - Example:
     “Great, I have you booked for Monday, December 15th at 10:00 AM Eastern, John Smith from Apex Plumbing, and your best number is 555-123-4567.”

6) If they do NOT want to pick a time:
   - Still collect name, phone, business name if possible.
   - Summarize what you captured and say that someone will follow up.

Strict rules:
- You MUST use `get_available_slots` whenever you need to know when the calendar is free.
- You MUST use `book_appointment` to actually schedule a consultation. Do NOT pretend to book something without calling this tool.
- Do NOT book over conflicts; if `book_appointment` says it failed, pick another slot.
- Keep all spoken responses short: 1–3 simple sentences.
- Never say you are an AI, a model, or mention prompts, tokens, APIs, Google Calendar, or internal tools or systems.
- Do not read raw JSON to the caller. Turn tool results into natural language.

Silence handling:
- If the caller is silent for a while, prompt once in a short, polite sentence.
- If there is still silence, end the call politely.

At the end of the conversation:
- Briefly summarize out loud what you captured (name, phone, business, and scheduled time if any) in one short sentence.
"""

# Events from OpenAI worth logging
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

# =========================================================
# Google Calendar helpers
# =========================================================


def _init_calendar_service():
    global calendar_service
    if not GOOGLE_SERVICE_ACCOUNT_JSON or not GOOGLE_CALENDAR_ID:
        logger.warning("Google Calendar not fully configured (missing JSON or CALENDAR_ID).")
        calendar_service = None
        return

    try:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=GOOGLE_SCOPES
        )
        calendar_service = build("calendar", "v3", credentials=creds)
        logger.info("Google Calendar service initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Google Calendar service: %s", e)
        calendar_service = None


def get_calendar_service():
    if calendar_service is None:
        logger.warning("Calendar service is not initialized or failed previously.")
    return calendar_service


def get_free_slots_for_date(
    target_date: date,
    slot_minutes: int = BOOKING_SLOT_MINUTES,
    max_slots: int = 5,
) -> List[Dict]:
    """
    Return up to `max_slots` free appointment slots for the given date.
    Each slot is { "start": ISO8601, "end": ISO8601 } in LOCAL_TZ.
    """
    service = get_calendar_service()
    if not service:
        logger.warning("get_free_slots_for_date: calendar service is None.")
        return []

    day_start = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=BOOKING_WORK_START_HOUR,
        minute=0,
        second=0,
        tzinfo=LOCAL_TZ,
    )
    day_end = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=BOOKING_WORK_END_HOUR,
        minute=0,
        second=0,
        tzinfo=LOCAL_TZ,
    )

    body = {
        "timeMin": day_start.isoformat(),
        "timeMax": day_end.isoformat(),
        "items": [{"id": GOOGLE_CALENDAR_ID}],
    }

    try:
        fb = service.freebusy().query(body=body).execute()
        busy_list = fb["calendars"][GOOGLE_CALENDAR_ID]["busy"]
    except Exception as e:
        logger.exception("Error calling freeBusy: %s", e)
        return []

    busy_intervals: List[Tuple[datetime, datetime]] = []
    for b in busy_list:
        bs = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        be = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        busy_intervals.append((bs, be))

    busy_intervals.sort(key=lambda x: x[0])

    free_intervals: List[Tuple[datetime, datetime]] = []
    cursor = day_start

    for bs, be in busy_intervals:
        if be <= day_start or bs >= day_end:
            continue

        bs_clamped = max(bs, day_start)
        be_clamped = min(be, day_end)

        if bs_clamped > cursor:
            free_intervals.append((cursor, bs_clamped))

        cursor = max(cursor, be_clamped)

    if cursor < day_end:
        free_intervals.append((cursor, day_end))

    slots: List[Dict] = []
    slot_delta = timedelta(minutes=slot_minutes)

    for fs, fe in free_intervals:
        slot_start = fs
        while slot_start + slot_delta <= fe and len(slots) < max_slots:
            slot_end = slot_start + slot_delta
            slots.append(
                {
                    "start": slot_start.isoformat(),
                    "end": slot_end.isoformat(),
                }
            )
            slot_start = slot_end
        if len(slots) >= max_slots:
            break

    logger.info("Computed %d free slots for %s", len(slots), target_date.isoformat())
    return slots


def find_next_free_slot(duration_minutes: int = BOOKING_SLOT_MINUTES) -> Optional[Tuple[datetime, datetime]]:
    """
    Look ahead BOOKING_LOOKAHEAD_DAYS from now and return
    the earliest free slot that fits `duration_minutes`.
    """
    now = datetime.now(LOCAL_TZ)
    for day_offset in range(BOOKING_LOOKAHEAD_DAYS + 1):
        d = (now + timedelta(days=day_offset)).date()
        slots = get_free_slots_for_date(d, duration_minutes, max_slots=100)
        for s in slots:
            start = datetime.fromisoformat(s["start"])
            end = datetime.fromisoformat(s["end"])
            if start >= now:
                return start, end
    return None


def create_calendar_event(
    summary: str,
    start_dt: datetime,
    end_dt: datetime,
    description: str = "",
) -> Optional[Dict]:
    """
    Create a calendar event ONLY if the requested time is free.
    Returns the event dict on success, or None if:
    - calendar is not configured, or
    - the time range conflicts with existing events, or
    - the insert fails.
    """
    service = get_calendar_service()
    if not service:
        logger.warning("create_calendar_event: calendar service is None.")
        return None

    # 1) Check for conflicts using FreeBusy
    try:
        fb_body = {
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "items": [{"id": GOOGLE_CALENDAR_ID}],
        }
        fb = service.freebusy().query(body=fb_body).execute()
        busy = fb["calendars"][GOOGLE_CALENDAR_ID]["busy"]

        if busy:
            logger.warning(
                "create_calendar_event: conflict detected in [%s, %s], busy=%s",
                start_dt.isoformat(),
                end_dt.isoformat(),
                busy,
            )
            return None  # do NOT book over conflicts
    except Exception as e:
        logger.exception("create_calendar_event: FreeBusy conflict check failed: %s", e)
        # Fail closed: if we can’t verify, do not book
        return None

    # 2) If no conflicts, insert the event
    body = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": TIMEZONE_STR,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": TIMEZONE_STR,
        },
    }

    try:
        event = service.events().insert(
            calendarId=GOOGLE_CALENDAR_ID,
            body=body,
            sendUpdates="none",
        ).execute()
        logger.info("Created calendar event: %s", event.get("id"))
        return event
    except Exception as e:
        logger.exception("Failed to create calendar event: %s", e)
        return None


# =========================================================
# Tool wrappers for the model
# =========================================================

def tool_get_available_slots(max_slots: int = 5) -> Dict:
    """
    Tool wrapper:
    Returns up to `max_slots` upcoming free slots over the next BOOKING_LOOKAHEAD_DAYS.
    """
    now = datetime.now(LOCAL_TZ)
    slots: List[Dict] = []

    for day_offset in range(BOOKING_LOOKAHEAD_DAYS + 1):
        d = (now + timedelta(days=day_offset)).date()
        remaining = max_slots - len(slots)
        if remaining <= 0:
            break

        day_slots = get_free_slots_for_date(
            target_date=d,
            slot_minutes=BOOKING_SLOT_MINUTES,
            max_slots=remaining,
        )
        slots.extend(day_slots)

        if len(slots) >= max_slots:
            break

    return {
        "ok": True,
        "slots": slots,
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
    """
    Tool wrapper:
    Attempts to book an appointment and records the lead.
    """
    try:
        start_dt = datetime.fromisoformat(slot_start)
        end_dt = datetime.fromisoformat(slot_end)

        # Ensure times are timezone-aware
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=LOCAL_TZ)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=LOCAL_TZ)

        description_parts = [
            f"Caller: {caller_name}",
            f"Phone: {caller_phone}",
            f"Business: {business_name}",
        ]
        if notes:
            description_parts.append(f"Notes: {notes}")
        description = " | ".join(description_parts)

        event = create_calendar_event(
            summary=f"{BUSINESS_NAME} consultation - {business_name}",
            start_dt=start_dt,
            end_dt=end_dt,
            description=description,
        )

        if not event:
            return {"ok": False, "error": "event_create_failed_or_conflict"}

        # Log lead into in-memory inbox
        LEADS.append(
            {
                "created_at": datetime.now(LOCAL_TZ).isoformat(),
                "caller_name": caller_name,
                "caller_phone": caller_phone,
                "business_name": business_name,
                "slot_start": start_dt.isoformat(),
                "slot_end": end_dt.isoformat(),
                "notes": notes,
                "event_id": event.get("id"),
            }
        )

        safe_event = {
            "id": event.get("id"),
            "summary": event.get("summary"),
            "start": event.get("start"),
            "end": event.get("end"),
            "htmlLink": event.get("htmlLink"),
        }

        return {"ok": True, "event": safe_event}

    except Exception as e:
        logger.exception("tool_book_appointment failed: %s", e)
        return {"ok": False, "error": str(e)}


# =========================================================
# OpenAI Realtime helpers
# =========================================================

async def configure_openai_session(openai_ws):
    """
    Send session.update and initial greeting to OpenAI Realtime,
    including tool definitions for calendar availability and booking.
    """
    session_update = {
        "type": "session.update",
        "session": {
            "model": OPENAI_REALTIME_MODEL,
            "modalities": ["audio", "text"],
            "voice": OPENAI_VOICE,
            "instructions": SYSTEM_INSTRUCTIONS.strip(),
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
            "tools": [
                {
                    "type": "function",
                    "name": "get_available_slots",
                    "description": "Get upcoming free 15-minute slots from the owner’s calendar over the next few days.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_slots": {
                                "type": "integer",
                                "description": "Maximum number of slots to return (1–20).",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5,
                            }
                        },
                        "required": [],
                    },
                },
                {
                    "type": "function",
                    "name": "book_appointment",
                    "description": (
                        "Book a 15-minute consultation in the owner’s calendar and record the lead. "
                        "You must pass the chosen slot start/end along with caller name, phone, and business name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot_start": {
                                "type": "string",
                                "format": "date-time",
                                "description": "The selected slot start time in ISO 8601 (e.g. 2025-12-10T15:00:00-05:00).",
                            },
                            "slot_end": {
                                "type": "string",
                                "format": "date-time",
                                "description": "The selected slot end time in ISO 8601.",
                            },
                            "caller_name": {
                                "type": "string",
                                "description": "Caller’s full name.",
                            },
                            "caller_phone": {
                                "type": "string",
                                "description": "Caller’s best callback phone number.",
                            },
                            "business_name": {
                                "type": "string",
                                "description": "Caller’s business name.",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional short note about the call.",
                            },
                        },
                        "required": ["slot_start", "slot_end", "caller_name", "caller_phone", "business_name"],
                    },
                },
            ],
            "tool_choice": "auto",
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI")

    # Kick off with a greeting
    await openai_ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "instructions": (
                        f"Start the call by briefly introducing yourself as the receptionist for {BUSINESS_NAME} "
                        f"and ask how you can help."
                    )
                },
            }
        )
    )
    logger.info("Sent initial greeting request to OpenAI")


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    _init_calendar_service()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "BookingBeaver AI Receptionist",
        "model": OPENAI_REALTIME_MODEL,
        "host": PUBLIC_HOST,
        "business": BUSINESS_NAME,
        "voice": OPENAI_VOICE,
    }


@app.get("/debug/env")
async def debug_env():
    return {
        "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
        "OPENAI_REALTIME_MODEL": OPENAI_REALTIME_MODEL,
        "PUBLIC_HOST": PUBLIC_HOST,
        "BUSINESS_NAME": BUSINESS_NAME,
        "VOICE": OPENAI_VOICE,
        "LOG_LEVEL": LOG_LEVEL,
        "GOOGLE_CALENDAR_CONFIGURED": bool(GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_CALENDAR_ID),
        "GOOGLE_CALENDAR_ID": GOOGLE_CALENDAR_ID,
        "TIMEZONE": TIMEZONE_STR,
        "BOOKING_SLOT_MINUTES": BOOKING_SLOT_MINUTES,
        "BOOKING_WORK_START_HOUR": BOOKING_WORK_START_HOUR,
        "BOOKING_WORK_END_HOUR": BOOKING_WORK_END_HOUR,
        "BOOKING_LOOKAHEAD_DAYS": BOOKING_LOOKAHEAD_DAYS,
    }


# =========================================================
# Twilio Voice Webhook
# =========================================================

@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """
    Twilio webhook: returns TwiML that starts a Media Stream to /media-stream.
    """
    form = await request.form()
    from_number = form.get("From", "unknown")
    to_number = form.get("To", "unknown")
    call_sid = form.get("CallSid", "unknown")

    logger.info(
        "Incoming call from Twilio: from=%s to=%s call_sid=%s",
        from_number,
        to_number,
        call_sid,
    )

    stream_url = f"wss://{PUBLIC_HOST}/media-stream"
    logger.info("TwiML connect stream URL: %s", stream_url)

    vr = VoiceResponse()
    vr.say(
        f"Connecting you to the {BUSINESS_NAME} receptionist.",
        voice="Polly.Joanna",
    )

    connect = Connect()
    connect.append(Stream(url=stream_url))
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


# =========================================================
# WebSocket Bridge: Twilio <-> OpenAI Realtime
# =========================================================

async def openai_session_bridge(websocket_twilio: WebSocket):
    """
    Bridge audio between Twilio Media Streams WebSocket and OpenAI Realtime WebSocket.
    """
    realtime_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info("Connecting to OpenAI Realtime at %s", realtime_url)

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
                        start_info = msg.get("start", {})
                        stream_sid = start_info.get("streamSid")
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
                        await websocket_openai.send(
                            json.dumps({"type": "input_audio_buffer.commit"})
                        )
                        break

            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
            except Exception as e:
                logger.exception("Error in from_twilio_to_openai: %s", e)

        async def from_openai_to_twilio():
            nonlocal stream_sid
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

                    # Audio stream back to Twilio
                    if evt_type == "response.audio.delta":
                        if not stream_sid:
                            logger.warning("Received audio.delta before stream_sid is set")
                            continue
                        delta_b64 = evt.get("delta")
                        if not delta_b64:
                            continue

                        twilio_msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": delta_b64,
                            },
                        }
                        await websocket_twilio.send_text(json.dumps(twilio_msg))
                        continue

                    # Tool calls from the model
                    if evt_type == "response.output_item.added":
                        item = evt.get("item", {})
                        if item.get("type") != "tool_call":
                            continue

                        tool_name = item.get("name")
                        tool_call_id = item.get("id")
                        args_raw = item.get("arguments") or {}

                        # arguments may arrive as a JSON string – decode if needed
                        if isinstance(args_raw, str):
                            try:
                                args = json.loads(args_raw)
                            except Exception:
                                args = {}
                        else:
                            args = args_raw

                        logger.info("Tool call requested: %s -> %s", tool_name, args)

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
                            logger.exception("Error executing tool %s: %s", tool_name, tool_err)
                            result = {"ok": False, "error": str(tool_err)}

                        # Send tool result back to OpenAI
                        try:
                            await websocket_openai.send(
                                json.dumps(
                                    {
                                        "type": "input_tool_output",
                                        "tool_output": {
                                            "tool_call_id": tool_call_id,
                                            "output": result,
                                        },
                                    }
                                )
                            )
                            logger.info("Sent tool result for %s", tool_name)
                        except Exception as send_err:
                            logger.exception("Failed to send tool result: %s", send_err)

                        continue

            except ConnectionClosed:
                logger.warning("OpenAI websocket closed")
            except Exception as e:
                logger.exception("Error in from_openai_to_twilio: %s", e)

        await asyncio.gather(from_twilio_to_openai(), from_openai_to_twilio())
        logger.info("OpenAI session bridge finished")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio Media Streams connects here.
    """
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket connected")
    try:
        await openai_session_bridge(websocket)
    finally:
        logger.info("Closing Twilio Media Stream WebSocket")
        try:
            await websocket.close()
        except Exception:
            pass


# =========================================================
# Debug / Lead endpoints
# =========================================================

@app.get("/debug/calendar")
async def debug_calendar():
    """
    Return some upcoming events to confirm calendar access works.
    """
    service = get_calendar_service()
    if not service:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "calendar_not_configured"},
        )

    try:
        now = datetime.now(LOCAL_TZ)
        events_result = (
            service.events()
            .list(
                calendarId=GOOGLE_CALENDAR_ID,
                timeMin=now.isoformat(),
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
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "calendar_error", "detail": str(e)},
        )


@app.api_route("/debug/book-test", methods=["GET", "POST"])
async def debug_book_test():
    """
    Create a 15-minute test event using the next free slot.
    """
    try:
        slot = find_next_free_slot(BOOKING_SLOT_MINUTES)
        if not slot:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "no_free_slot"},
            )

        start, end = slot
        event = create_calendar_event(
            summary="Test BookingBeaver AI appointment",
            start_dt=start,
            end_dt=end,
            description="Debug test booking via /debug/book-test",
        )
        if not event:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "event_create_failed_or_conflict"},
            )

        safe_event = {
            "id": event.get("id"),
            "summary": event.get("summary"),
            "start": event.get("start"),
            "end": event.get("end"),
            "htmlLink": event.get("htmlLink"),
        }
        return {"ok": True, "event": safe_event}

    except Exception as e:
        logger.exception("Error in /debug/book-test: %s", e)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "debug_book_test_failed", "detail": str(e)},
        )


@app.get("/debug/free-slots")
async def debug_free_slots(
    dt_str: Optional[str] = Query(default=None, description="YYYY-MM-DD (optional)"),
    max_slots: int = Query(default=5, ge=1, le=50),
):
    """
    Return free slots for a given date (or today if not provided).
    """
    try:
        if dt_str:
            target_date = date.fromisoformat(dt_str)
        else:
            target_date = datetime.now(LOCAL_TZ).date()

        slots = get_free_slots_for_date(
            target_date=target_date,
            slot_minutes=BOOKING_SLOT_MINUTES,
            max_slots=max_slots,
        )
        return {
            "ok": True,
            "date": target_date.isoformat(),
            "slot_minutes": BOOKING_SLOT_MINUTES,
            "slots": slots,
        }
    except Exception as e:
        logger.exception("Error in /debug/free-slots: %s", e)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "debug_free_slots_failed", "detail": str(e)},
        )


@app.get("/leads")
async def list_leads():
    """
    Lead inbox endpoint (in-memory).
    """
    return {"count": len(LEADS), "leads": LEADS}
