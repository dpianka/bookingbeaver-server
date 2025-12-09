import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import websockets

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(LOG_LEVEL, LOG_LEVEL, logging.INFO),
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
    "gpt-4o-realtime-preview",  # keep whatever is working for you now
)

PUBLIC_HOST = os.getenv("PUBLIC_HOST")  # e.g. bookingbeaver-server-production.up.railway.app
if not PUBLIC_HOST:
    raise RuntimeError("PUBLIC_HOST is not set (e.g. bookingbeaver-server-production.up.railway.app)")

# This is BookingBeaver’s own answering service POC
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "BookingBeaver")
VOICE = os.getenv("OPENAI_VOICE", "alloy")

# Optional Cal.com booking URL for the 15-min consult (spoken by the agent)
CALCOM_URL = os.getenv("CALCOM_URL", "https://cal.com/bookingbeaver/15min")

# In-memory lead inbox (POC – reset on redeploy)
CALLER_INDEX = {}  # call_sid -> {"from": "+1...", "to": "+1..."}
LEADS = []        # list of call_context dicts


SYSTEM_MESSAGE = f"""
You are the phone receptionist for BookingBeaver.

BookingBeaver is an AI answering service for service businesses. It:
- answers calls 24/7,
- answers common questions about the business,
- captures leads,
- and helps book appointments using Cal.com links.

Your job on each call:

1) Greet the caller warmly but briefly as BookingBeaver.
2) Ask how you can help.
3) If they are interested in BookingBeaver for their business:
   - Explain in simple, concrete terms what BookingBeaver can do:
     AI receptionist, answers calls, screens leads, books via Cal.com, sends messages to the owner.
   - Offer a free 15-minute consultation call.
   - Ask for:
       • Full name
       • Best callback phone number
       • Email address
       • Business name
       • Business type (for example: HVAC, plumbing, salon, law firm, etc.)
   - Ask for their preferred time window for the 15-minute consult and one backup time.
   - Clearly repeat these details back to them in a short confirmation sentence.
   - Tell them you’ll send them a Cal.com link (for example: {CALCOM_URL}) where they can confirm the exact time.

4) If they are not ready to book a consult:
   - Still politely ask for their name, phone, email, business name, and business type so someone from BookingBeaver can follow up.
   - Keep it light and low pressure.

5) If they ask “What can BookingBeaver provide?” or similar:
   - Explain BookingBeaver as an AI answering service:
     • It picks up their business calls,
     • answers common questions,
     • captures caller details,
     • and helps schedule jobs or consultations using a booking link like Cal.com.
   - Give 1–2 specific examples, not a long speech.

General rules:
- Keep replies short and clear for phone audio. Prefer 1–3 short sentences per turn.
- Do NOT mention that you are an AI, a model, or talk about tokens, prompts, or APIs.
- Do NOT read URLs slowly like a robot. Just mention “I’ll send you a Cal.com link to pick a 15-minute slot.”
- If the caller goes quiet for several seconds, gently prompt once. If they stay silent, wrap up politely.

At the end of the call, you should always have:
- Caller name
- Phone number
- Email
- Business name
- Business type
- Whether they want a 15-minute consult or just info.

At the very end of the conversation, summarize out loud in one short sentence what you captured, so a human can understand it later.
"""

LOG_EVENT_TYPES = [
    "session.created",
    "session.updated",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.created",
    "response.audio.delta",
    "response.done",
    "error",
]

app = FastAPI()


# ---------------------------------------------------------
# Helpers: OpenAI Realtime session config
# ---------------------------------------------------------
async def send_session_update(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "model": OPENAI_REALTIME_MODEL,
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE.strip(),
            "modalities": ["text", "audio"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 600,
                "threshold": 0.5,
                "prefix_padding_ms": 300,
            },
        },
    }
    logger.info("Sending session.update to OpenAI")
    await openai_ws.send(json.dumps(session_update))


async def send_initial_greeting(openai_ws):
    item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"A caller has just been connected. Greet them as the phone receptionist for {BUSINESS_NAME}. "
                        "Briefly say who you are, that you help explain BookingBeaver and can set up a 15-minute consultation, "
                        "then ask how you can help today."
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
    We:
      - parse the form body for From/To/CallSid,
      - store caller phone in CALLER_INDEX for lead inbox,
      - return TwiML that says a short line and starts the media stream.
    """
    body_bytes = await request.body()
    try:
        params = parse_qs(body_bytes.decode())
    except Exception:
        params = {}

    from_number = (params.get("From") or ["unknown"])[0]
    to_number = (params.get("To") or ["unknown"])[0]
    call_sid = (params.get("CallSid") or ["unknown"])[0]

    CALLER_INDEX[call_sid] = {
        "from": from_number,
        "to": to_number,
    }

    logger.info(
        "Incoming call from Twilio: from=%s to=%s call_sid=%s",
        from_number,
        to_number,
        call_sid,
    )

    vr = VoiceResponse()
    vr.say(
        "Connecting you to the BookingBeaver AI receptionist.",
        voice="Polly.Joanna",
    )

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
      - Store a basic call record in LEADS for the lead inbox
    """
    await websocket.accept(subprotocol="audio.twilio.com")
    logger.info("Twilio Media Stream WebSocket connected")

    openai_ws = None

    call_context = {
        "stream_sid": None,
        "call_sid": None,
        "account_sid": None,
        "caller_phone": None,
        "called_number": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "model": OPENAI_REALTIME_MODEL,
        "voice": VOICE,
        "business": BUSINESS_NAME,
    }

    try:
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
            nonlocal stream_sid, call_context
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event in ("start", "connected", "stop", "mark"):
                        logger.info("From Twilio: %s -> %s", event, data)

                    if event == "start":
                        start_info = data.get("start", {})
                        stream_sid = start_info.get("streamSid")
                        call_sid = start_info.get("callSid")
                        call_context["stream_sid"] = stream_sid
                        call_context["call_sid"] = call_sid
                        call_context["account_sid"] = start_info.get("accountSid")

                        # Attach caller phone / called number from the initial webhook, if we have it
                        caller_meta = CALLER_INDEX.get(call_sid) or {}
                        call_context["caller_phone"] = caller_meta.get("from")
                        call_context["called_number"] = caller_meta.get("to")

                        logger.info("Twilio stream started: %s (call_sid=%s)", stream_sid, call_sid)

                    elif event == "media":
                        payload_b64 = data["media"]["payload"]
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": payload_b64,
                        }
                        await openai_ws.send(json.dumps(audio_append))

                    elif event == "stop":
                        logger.info("Twilio stream stopped event received")
                        break

                    else:
                        if event not in (None, "start", "connected", "media", "stop"):
                            logger.info("From Twilio (other event): %s -> %s", event, data)

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
                        try:
                            await websocket.send_json(audio_delta)
                        except Exception as send_err:
                            logger.exception("Error sending audio delta to Twilio: %s", send_err)
                            break

            except websockets.ConnectionClosed:
                logger.warning("OpenAI websocket closed")
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

        call_context["ended_at"] = datetime.now(timezone.utc).isoformat()
        LEADS.append(call_context)

        logger.info("CALL_LEAD %s", json.dumps(call_context))
        logger.info("Closing Twilio Media Stream WebSocket")
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------
# Health + debug + lead inbox
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "BookingBeaver AI Receptionist (Realtime)",
        "model": OPENAI_REALTIME_MODEL,
        "host": PUBLIC_HOST,
        "business": BUSINESS_NAME,
        "voice": VOICE,
    }


@app.get("/debug/env")
async def debug_env():
    return JSONResponse(
        {
            "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
            "OPENAI_REALTIME_MODEL": OPENAI_REALTIME_MODEL,
            "PUBLIC_HOST": PUBLIC_HOST,
            "BUSINESS_NAME": BUSINESS_NAME,
            "VOICE": VOICE,
            "LOG_LEVEL": LOG_LEVEL,
        }
    )


@app.get("/leads")
async def list_leads():
    """
    Simple lead inbox endpoint (POC).

    Returns a list of call records:
    - caller_phone
    - called_number
    - call_sid
    - timestamps
    - model / voice / business

    This is enough to prove that:
    - calls are answered by AI,
    - leads are being warehoused centrally.
    """
    return {"count": len(LEADS), "leads": LEADS}
