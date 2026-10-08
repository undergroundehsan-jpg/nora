import asyncio
import base64
import json
import os
import struct
import time
import uuid
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

try:
    import audioop
except ModuleNotFoundError:
    # audioop was removed in Python 3.13; audioop-lts is the drop-in replacement.
    import audioop_lts as audioop  # type: ignore[no-redef]

import numpy as np
import websockets
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from dotenv import load_dotenv

load_dotenv()

twilio_router = APIRouter()

# Populated by main.py at startup via register_session_store().
# Used by the status callback to clean up completed call sessions.
_session_store: dict | None = None


def register_session_store(store: dict) -> None:
    """Call once from main.py after session_store is created."""
    global _session_store
    _session_store = store


def get_twilio_provider_status() -> dict:
    """Return Twilio provider configuration health for diagnostics."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_NUMBER", "").strip()
    public_base = os.getenv("PUBLIC_BASE_URL", "").strip()
    enabled = os.getenv("TWILIO_ENABLED", "true").strip().lower() not in {"0", "false", "no"}

    return {
        "enabled": enabled,
        "configured": bool(account_sid and auth_token and from_number and public_base),
        "has_default_to": bool(os.getenv("TWILIO_OUTBOUND_DEFAULT_TO", "").strip()),
    }


def _twilio_is_enabled() -> bool:
    return os.getenv("TWILIO_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


async def check_twilio_connectivity() -> dict:
    """Lightweight Twilio auth/connectivity check used by diagnostics endpoints."""
    if not _twilio_is_enabled():
        return {"ok": False, "error": "Twilio provider is disabled (TWILIO_ENABLED=false)"}

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not account_sid or not auth_token:
        return {"ok": False, "error": "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set"}

    try:
        from twilio.rest import Client as TwilioClient  # type: ignore[import-untyped]
        from twilio.http.http_client import TwilioHttpClient  # type: ignore[import-untyped]

        http_client = TwilioHttpClient(timeout=8)
        client = TwilioClient(account_sid, auth_token, http_client=http_client)

        def _fetch_account():
            return client.api.accounts(account_sid).fetch()

        account = await asyncio.wait_for(asyncio.to_thread(_fetch_account), timeout=10)
        return {
            "ok": True,
            "account_sid": account_sid,
            "account_status": getattr(account, "status", "unknown"),
            "friendly_name": getattr(account, "friendly_name", ""),
        }
    except asyncio.TimeoutError:
        return {"ok": False, "error": "Twilio connectivity check timed out"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _validate_twilio_signature(request: Request, params: dict[str, str]) -> bool:
    """
    Validate that an HTTP webhook request originated from Twilio.
    Returns True when validation passes or when TWILIO_AUTH_TOKEN is not set
    (treats missing token as dev mode — validation skipped with a warning).
    """
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not auth_token:
        print("[TWILIO] WARNING: TWILIO_AUTH_TOKEN not set — skipping signature validation (dev mode).")
        return True

    try:
        from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]
    except ImportError:
        print("[TWILIO] WARNING: twilio package not installed — skipping signature validation.")
        return True

    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(auth_token)

    # Twilio signs the externally visible webhook URL. Behind Railway/reverse
    # proxies, request.url may appear as http://internal-host which fails checks.
    raw_url = str(request.url)
    parsed = urlparse(raw_url)
    path_and_query = parsed.path + (f"?{parsed.query}" if parsed.query else "")

    forwarded_host = request.headers.get("x-forwarded-host", "").strip()
    forwarded_proto = request.headers.get("x-forwarded-proto", "").strip()
    host = request.headers.get("host", "").strip()
    public_base = _normalize_public_base_url(os.getenv("PUBLIC_BASE_URL", ""))

    candidates: list[str] = [raw_url]
    if forwarded_host and forwarded_proto:
        candidates.append(f"{forwarded_proto}://{forwarded_host}{path_and_query}")
    if host:
        candidates.append(f"https://{host}{path_and_query}")
        candidates.append(f"http://{host}{path_and_query}")
    if public_base:
        candidates.append(f"{public_base}{path_and_query}")

    # Preserve order while removing duplicates.
    deduped_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped_candidates:
            deduped_candidates.append(candidate)

    for candidate_url in deduped_candidates:
        try:
            if validator.validate(candidate_url, params, signature):
                return True
        except Exception:
            continue

    print(
        "[TWILIO] Signature validation FAILED "
        f"raw_url={raw_url} "
        f"forwarded_proto={forwarded_proto or 'n/a'} "
        f"forwarded_host={forwarded_host or 'n/a'}"
    )
    return False


def _append_query(url: str, extra_params: dict[str, str]) -> str:
    parsed = urlparse(url)
    existing = parse_qs(parsed.query, keep_blank_values=True)
    for key, value in extra_params.items():
        existing[key] = [value]
    query = urlencode(existing, doseq=True)
    return urlunparse(parsed._replace(query=query))


def _normalize_public_base_url(raw_value: str) -> str:
    """Return normalized absolute base URL; add https:// when scheme is omitted."""
    base = (raw_value or "").strip().rstrip("/")
    if not base:
        return ""

    if "//" not in base:
        base = f"https://{base}"

    parsed = urlparse(base)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _build_media_stream_ws_url(request: Request, session_id: str, call_sid: str, lead_id: str = "") -> str:
    explicit_ws = os.getenv("TWILIO_MEDIA_WS_URL", "").strip()
    if explicit_ws:
        base = explicit_ws
    else:
        public_base = _normalize_public_base_url(os.getenv("PUBLIC_BASE_URL", ""))
        if public_base:
            parsed = urlparse(public_base)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            base = f"{ws_scheme}://{parsed.netloc}/twilio/media"
        else:
            req_url = request.url
            ws_scheme = "wss" if req_url.scheme == "https" else "ws"
            base = f"{ws_scheme}://{req_url.netloc}/twilio/media"

    params = {"session_id": session_id, "call_sid": call_sid}
    if lead_id:
        params["lead_id"] = lead_id
    return _append_query(base, params)


def _build_agent_ws_url(session_id: str, lead_id: str = "") -> str:
    # Railway sets PORT (typically 8080). Local dev often runs on 8000.
    default_port = os.getenv("PORT", "8000").strip() or "8000"
    default_base = f"ws://127.0.0.1:{default_port}/ws/voice"
    base = os.getenv("TWILIO_AGENT_WS_URL", default_base).strip()
    params = {"session_id": session_id}
    if lead_id:
        params["lead_id"] = lead_id
    return _append_query(base, params)


def _ulaw_8k_to_pcm16_16k(payload_b64: str) -> bytes:
    try:
        mulaw_bytes = base64.b64decode(payload_b64)
    except Exception:
        return b""

    # Twilio media stream payload is 8 kHz mu-law mono.
    pcm8 = audioop.ulaw2lin(mulaw_bytes, 2)
    if not pcm8:
        return b""

    samples_8k = np.frombuffer(pcm8, dtype=np.int16)
    if samples_8k.size == 0:
        return b""

    # Lightweight 2x upsample to 16 kHz for the existing /ws/voice pipeline.
    samples_16k = np.repeat(samples_8k, 2)
    return samples_16k.astype(np.int16).tobytes()


def _pcm16_16k_to_ulaw_8k(pcm16_16k: bytes) -> bytes:
    if not pcm16_16k:
        return b""

    samples_16k = np.frombuffer(pcm16_16k, dtype=np.int16)
    if samples_16k.size == 0:
        return b""

    # 2:1 downsample to 8 kHz before mu-law encoding for Twilio.
    samples_8k = samples_16k[::2]
    pcm8 = samples_8k.astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm8, 2)


async def _extract_request_params(request: Request) -> dict[str, str]:
    if request.method.upper() == "GET":
        return dict(request.query_params)

    body = (await request.body()).decode("utf-8", errors="ignore")
    parsed = parse_qs(body, keep_blank_values=True)
    out = {}
    for key, values in parsed.items():
        out[key] = values[0] if values else ""
    return out


@twilio_router.api_route("/voice/inbound", methods=["GET", "POST"])
async def twilio_voice_inbound(request: Request) -> Response:
    params = await _extract_request_params(request)

    if not _validate_twilio_signature(request, params):
        return Response(content="Forbidden", status_code=403)

    call_sid = params.get("CallSid", "")
    session_id = call_sid or str(uuid.uuid4())
    # Twilio fetches the exact URL we gave it, so the lead id comes back here.
    lead_id = (params.get("lead_id") or request.query_params.get("lead_id") or "").strip()

    stream_url = _build_media_stream_ws_url(
        request, session_id=session_id, call_sid=call_sid, lead_id=lead_id
    )
    stream_url_xml = escape(stream_url, {'"': "&quot;"})
    preconnect_say = os.getenv("TWILIO_PRECONNECT_SAY", "").strip()
    preconnect_xml = ""
    if preconnect_say:
        # Optional immediate prompt to avoid dead air while media stream and agent warm up.
        safe_text = escape(preconnect_say)
        preconnect_xml = f"<Say>{safe_text}</Say>"

    twiml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        f"{preconnect_xml}"
        "<Connect>"
        f"<Stream url=\"{stream_url_xml}\" />"
        "</Connect>"
        "<Hangup/>"
        "</Response>"
    )

    print(f"[TWILIO] Inbound call mapped call_sid={call_sid or 'unknown'} session_id={session_id} "
          f"lead_id={lead_id or 'none'}")
    return Response(content=twiml, media_type="application/xml")


# Terminal statuses sent by Twilio in the status callback.
_TERMINAL_CALL_STATUSES = {"completed", "failed", "busy", "no-answer", "canceled"}


@twilio_router.api_route("/voice/status", methods=["GET", "POST"])
async def twilio_voice_status(request: Request) -> dict:
    params = await _extract_request_params(request)

    if not _validate_twilio_signature(request, params):
        return Response(content="Forbidden", status_code=403)

    call_sid = params.get("CallSid", "")
    call_status = params.get("CallStatus", "")
    to_number = params.get("To", "")
    from_number = params.get("From", "")
    sip_response_code = params.get("SipResponseCode", "")
    error_code = params.get("ErrorCode", "")
    error_message = params.get("ErrorMessage", "")
    call_duration = params.get("CallDuration", "")
    answered_by = params.get("AnsweredBy", "")

    print(
        "[TWILIO] Status callback "
        f"call_sid={call_sid or 'unknown'} "
        f"status={call_status or 'unknown'} "
        f"to={to_number or 'unknown'} "
        f"from={from_number or 'unknown'} "
        f"sip_code={sip_response_code or 'n/a'} "
        f"error_code={error_code or 'n/a'} "
        f"duration={call_duration or 'n/a'} "
        f"answered_by={answered_by or 'n/a'}"
    )
    if error_message:
        print(f"[TWILIO] Status detail error_message={error_message}")

    # Every attempt is logged, answered or not: unanswered calls never create a
    # session, so without this the dashboard could not count or retry them.
    try:
        from app.utils.logger import log_call_attempt
        await log_call_attempt({
            "call_sid": call_sid,
            "lead_id": (params.get("lead_id") or request.query_params.get("lead_id") or "") or None,
            "to": to_number,
            "from": from_number,
            "status": call_status,
            "duration": int(call_duration) if str(call_duration).isdigit() else None,
            "answered_by": answered_by or None,
            "error_code": error_code or None,
            "sip_response_code": sip_response_code or None,
        })
    except Exception:
        pass  # logging must never break a status callback

    if call_status in _TERMINAL_CALL_STATUSES and _session_store is not None:
        # The session key is set to call_sid in twilio_voice_inbound.
        session = _session_store.pop(call_sid, None)
        if session is not None:
            print(f"[TWILIO] Removed session from store for completed call call_sid={call_sid}")

    return {"ok": True}


def _build_inbound_webhook_url(lead_id: str = "") -> str:
    """Return the absolute HTTPS URL for /twilio/voice/inbound used as TwiML url in outbound calls."""
    public_base = _normalize_public_base_url(os.getenv("PUBLIC_BASE_URL", ""))
    if not public_base:
        raise ValueError(
            "PUBLIC_BASE_URL must be a valid URL (example: https://your-domain) "
            "(Twilio needs a reachable URL to fetch TwiML)."
        )
    url = f"{public_base}/twilio/voice/inbound"
    return _append_query(url, {"lead_id": lead_id}) if lead_id else url


async def place_twilio_outbound_call(payload: dict | None = None) -> tuple[dict, int]:
    """Place an outbound call via Twilio and return (payload, status_code)."""
    if not _twilio_is_enabled():
        return {"error": "Twilio provider is disabled (TWILIO_ENABLED=false)"}, 503

    try:
        from twilio.rest import Client as TwilioClient  # type: ignore[import-untyped]
    except ImportError:
        return {"error": "twilio package not installed - run pip install twilio"}, 500

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not account_sid or not auth_token:
        return {"error": "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in .env"}, 500

    body = payload or {}
    lead_id = str(body.get("lead_id") or "").strip()
    lead_phone = ""
    if lead_id:
        from app.services.leads import LeadError, lead_for_call, phone_for_lead
        try:
            lead = lead_for_call(lead_id)
        except LeadError as exc:
            return {"error": str(exc)}, 400
        lead_phone = phone_for_lead(lead)
        if not lead_phone and not body.get("to"):
            return {"error": f"Lead '{lead_id}' has no usable phone number"}, 400

    # Hard stop: never dial out from a test run, whatever the caller patched.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return {"error": "Outbound calls are disabled during tests"}, 503

    # Explicit "to" wins, then the lead's own number, then the .env default.
    to_number = body.get("to") or lead_phone or os.getenv("TWILIO_OUTBOUND_DEFAULT_TO", "").strip()

    # No lead was named (the "Make a Phone Call" button sends none), so match
    # the number being dialled against the customer list. Without this the call
    # runs as the sample customer, with no amount or due date to talk about.
    if not lead_id and to_number:
        from app.services.leads import list_leads, normalize_phone
        dialled = normalize_phone(to_number)
        for candidate in list_leads():
            if dialled and normalize_phone(candidate.get("phone", "")) == dialled:
                lead_id = candidate["lead_id"]
                print(f"[TWILIO] Matched {to_number} to customer {lead_id} "
                      f"({candidate.get('lead_name', '')})")
                break

    from_number = body.get("from") or os.getenv("TWILIO_NUMBER", "").strip()

    if not to_number:
        return {
            "error": "Provide 'to' in the request body or set TWILIO_OUTBOUND_DEFAULT_TO in .env"
        }, 400
    if not from_number:
        return {
            "error": "Provide 'from' in the request body or set TWILIO_NUMBER in .env"
        }, 400

    try:
        twiml_url = _build_inbound_webhook_url(lead_id)
    except ValueError as exc:
        return {"error": str(exc)}, 500

    public_base = _normalize_public_base_url(os.getenv("PUBLIC_BASE_URL", ""))
    status_callback_url = f"{public_base}/twilio/voice/status"
    if lead_id:
        status_callback_url = _append_query(status_callback_url, {"lead_id": lead_id})

    timeout_secs = 12.0
    try:
        from twilio.http.http_client import TwilioHttpClient  # type: ignore[import-untyped]

        # Use short HTTP timeout so Railway requests fail fast instead of hanging
        # until edge timeout produces a generic 502.
        http_client = TwilioHttpClient(timeout=10)
        client = TwilioClient(account_sid, auth_token, http_client=http_client)

        def _create_call():
            return client.calls.create(
                to=to_number,
                from_=from_number,
                url=twiml_url,
                status_callback=status_callback_url,
                status_callback_method="POST",
                status_callback_event=["initiated", "ringing", "answered", "completed"],
            )

        call = await asyncio.wait_for(asyncio.to_thread(_create_call), timeout=timeout_secs)
    except asyncio.TimeoutError:
        print(
            "[TWILIO] Outbound call create timed out "
            f"after {timeout_secs:.0f}s to={to_number} from={from_number}"
        )
        return {
            "error": "Twilio call create timed out",
            "hint": "Check Railway outbound connectivity and Twilio credentials.",
        }, 504
    except Exception as exc:
        print(f"[TWILIO] Outbound call failed: {exc}")
        return {"error": str(exc)}, 502

    print(f"[TWILIO] Outbound call placed to={to_number} from={from_number} "
          f"call_sid={call.sid} lead_id={lead_id or 'none'}")
    return {
        "ok": True,
        "provider": "twilio",
        "call_sid": call.sid,
        "to": to_number,
        "from": from_number,
        "lead_id": lead_id or None,
    }, 200


@twilio_router.post("/call/outbound")
async def twilio_outbound_call(request: Request) -> dict:
    """
    Place an outbound call via the Twilio REST API.

    JSON body (all fields optional):
        { "to": "+923215222468", "from": "+12603773867" }

    Falls back to TWILIO_OUTBOUND_DEFAULT_TO / TWILIO_NUMBER env vars when
    the body fields are absent.
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    result, status_code = await place_twilio_outbound_call(body)
    return JSONResponse(content=result, status_code=status_code)


async def _forward_agent_audio_to_twilio(
    agent_ws,
    twilio_ws: WebSocket,
    stream_state: dict,
    stream_ready_event: asyncio.Event,
) -> None:
    last_generation = None
    stats = stream_state.setdefault("stats", {})

    try:
        while True:
            if not stream_state.get("stream_sid"):
                # Twilio has not sent the "start" event yet. Wait so early
                # greeting audio is not dropped before a streamSid exists.
                await stream_ready_event.wait()

            frame = await agent_ws.recv()
            if not isinstance(frame, (bytes, bytearray)):
                continue
            if len(frame) <= 4:
                continue

            stream_sid = stream_state.get("stream_sid")
            if not stream_sid:
                # Race guard: if streamSid was cleared or not visible yet,
                # skip this frame instead of sending malformed Twilio events.
                continue

            generation = struct.unpack("<I", frame[:4])[0]
            pcm16_16k = bytes(frame[4:])
            ulaw_8k = _pcm16_16k_to_ulaw_8k(pcm16_16k)
            if not ulaw_8k:
                continue

            if last_generation is None:
                last_generation = generation
            elif generation != last_generation:
                # Flush queued outbound media on barge-in/interruption generation changes.
                await twilio_ws.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": stream_sid,
                }))
                stats["clear_events"] = int(stats.get("clear_events", 0)) + 1
                last_generation = generation

            payload = base64.b64encode(ulaw_8k).decode("ascii")
            await twilio_ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": payload,
                },
            }))

            stats["agent_media_frames_out"] = int(stats.get("agent_media_frames_out", 0)) + 1
            stats["agent_media_bytes_out"] = int(stats.get("agent_media_bytes_out", 0)) + len(ulaw_8k)
            if not stats.get("first_agent_media_at"):
                stats["first_agent_media_at"] = time.time()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[TWILIO] Agent->Twilio forward stopped: {e}")


async def _twilio_media_keepalive(
    twilio_ws: WebSocket,
    stream_state: dict,
    stop_event: asyncio.Event,
) -> None:
    """Send periodic mark events to keep media stream active and detectable."""
    interval_secs = 15.0
    try:
        interval_secs = max(5.0, float(os.getenv("TWILIO_MEDIA_HEARTBEAT_SECS", "15")))
    except ValueError:
        interval_secs = 15.0

    try:
        while not stop_event.is_set():
            await asyncio.sleep(interval_secs)
            if stop_event.is_set():
                break
            stream_sid = stream_state.get("stream_sid")
            if not stream_sid:
                continue
            # The call may have ended between ticks; sending then raises noisily.
            if twilio_ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                await twilio_ws.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "keepalive"},
                }))
            except Exception as e:
                if not stop_event.is_set():
                    print(f"[TWILIO] Keepalive send failed: {e}")
                break
    except asyncio.CancelledError:
        raise


@twilio_router.websocket("/media")
async def twilio_media_bridge(websocket: WebSocket) -> None:
    await websocket.accept()

    query = dict(websocket.query_params)
    session_id = query.get("session_id") or str(uuid.uuid4())
    call_sid = query.get("call_sid", "")
    lead_id = query.get("lead_id", "")

    agent_ws_url = _build_agent_ws_url(session_id=session_id, lead_id=lead_id)
    stream_state = {
        "stream_sid": "",
        "stats": {
            "opened_at": time.time(),
            "twilio_media_frames_in": 0,
            "twilio_media_bytes_in": 0,
            "agent_media_frames_out": 0,
            "agent_media_bytes_out": 0,
            "clear_events": 0,
            "first_twilio_media_at": None,
            "first_agent_media_at": None,
            "stream_started_at": None,
        },
    }
    stream_ready_event = asyncio.Event()
    keepalive_stop_event = asyncio.Event()

    print(f"[TWILIO] Media bridge opening session_id={session_id} call_sid={call_sid or 'unknown'}")

    try:
        async with websockets.connect(
            agent_ws_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as agent_ws:
            forward_task = asyncio.create_task(
                _forward_agent_audio_to_twilio(agent_ws, websocket, stream_state, stream_ready_event)
            )
            keepalive_task = asyncio.create_task(
                _twilio_media_keepalive(websocket, stream_state, keepalive_stop_event)
            )

            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    except asyncio.TimeoutError:
                        continue

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"[TWILIO] Ignored malformed media frame session_id={session_id}")
                        continue

                    event_type = event.get("event", "")

                    if event_type == "start":
                        start = event.get("start", {})
                        stream_state["stream_sid"] = start.get("streamSid", "")
                        if stream_state["stream_sid"]:
                            stream_ready_event.set()
                        call_sid = start.get("callSid", call_sid)
                        stream_state["stats"]["stream_started_at"] = time.time()
                        print(
                            f"[TWILIO] Stream started stream_sid={stream_state['stream_sid'] or 'unknown'} "
                            f"call_sid={call_sid or 'unknown'} session_id={session_id}"
                        )

                    elif event_type == "media":
                        payload = event.get("media", {}).get("payload", "")
                        if payload:
                            stream_state["stats"]["twilio_media_frames_in"] = int(
                                stream_state["stats"].get("twilio_media_frames_in", 0)
                            ) + 1
                            stream_state["stats"]["twilio_media_bytes_in"] = int(
                                stream_state["stats"].get("twilio_media_bytes_in", 0)
                            ) + len(payload)
                            if not stream_state["stats"].get("first_twilio_media_at"):
                                stream_state["stats"]["first_twilio_media_at"] = time.time()

                        pcm16_16k = _ulaw_8k_to_pcm16_16k(payload)
                        if pcm16_16k:
                            try:
                                await agent_ws.send(pcm16_16k)
                            except websockets.exceptions.ConnectionClosedOK:
                                print(
                                    f"[TWILIO] Agent WS closed normally session_id={session_id}; closing Twilio media stream."
                                )
                                try:
                                    await websocket.close(code=1000)
                                except Exception:
                                    pass
                                break
                            except websockets.exceptions.ConnectionClosed as e:
                                print(
                                    f"[TWILIO] Agent WS closed session_id={session_id}: {e}; closing Twilio media stream."
                                )
                                try:
                                    await websocket.close(code=1000)
                                except Exception:
                                    pass
                                break

                    elif event_type == "stop":
                        print(f"[TWILIO] Stream stop received session_id={session_id}")
                        break

                    elif event_type in {"connected", "mark", "dtmf"}:
                        # No-op for bridge. Logged only if needed later.
                        continue

            except WebSocketDisconnect as e:
                print(
                    f"[TWILIO] Client disconnected session_id={session_id} "
                    f"close_code={getattr(e, 'code', 'unknown')}"
                )
            finally:
                keepalive_stop_event.set()
                if not forward_task.done():
                    forward_task.cancel()
                    try:
                        await forward_task
                    except asyncio.CancelledError:
                        pass
                if not keepalive_task.done():
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass

    except Exception as e:
        print(f"[TWILIO] Media bridge error session_id={session_id}: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        stats = stream_state.get("stats", {})
        opened_at = float(stats.get("opened_at") or time.time())
        now = time.time()
        first_in = stats.get("first_twilio_media_at")
        first_out = stats.get("first_agent_media_at")
        print(
            "[TWILIO] Media diagnostics "
            f"session_id={session_id} "
            f"call_sid={call_sid or 'unknown'} "
            f"elapsed_s={now - opened_at:.2f} "
            f"in_frames={int(stats.get('twilio_media_frames_in', 0))} "
            f"in_bytes_b64={int(stats.get('twilio_media_bytes_in', 0))} "
            f"out_frames={int(stats.get('agent_media_frames_out', 0))} "
            f"out_bytes_ulaw={int(stats.get('agent_media_bytes_out', 0))} "
            f"clear_events={int(stats.get('clear_events', 0))} "
            f"first_in_after_s={((first_in - opened_at) if first_in else -1):.2f} "
            f"first_out_after_s={((first_out - opened_at) if first_out else -1):.2f}"
        )
        print(f"[TWILIO] Media bridge closed session_id={session_id}")
