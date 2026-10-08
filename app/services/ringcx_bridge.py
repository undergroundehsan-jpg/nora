import asyncio
import base64
import hashlib
import hmac
import json
import os
import struct
import uuid
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

try:
    import audioop
except ModuleNotFoundError:
    # audioop was removed in Python 3.13; audioop-lts is the drop-in replacement.
    import audioop_lts as audioop  # type: ignore[no-redef]

import httpx
import numpy as np
import websockets
from dotenv import load_dotenv
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

load_dotenv()

ringcx_router = APIRouter()

# Populated by main.py at startup via register_ringcx_session_store().
# Used by status callbacks to clean up completed call sessions.
_session_store: dict | None = None

_TERMINAL_RINGCX_STATUSES = {
    "completed",
    "complete",
    "ended",
    "terminated",
    "failed",
    "busy",
    "no-answer",
    "no_answer",
    "canceled",
    "cancelled",
    "hangup",
}


def register_ringcx_session_store(store: dict) -> None:
    global _session_store
    _session_store = store


def get_ringcx_provider_status() -> dict:
    enabled = os.getenv("RINGCX_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
    outbound_url = os.getenv("RINGCX_OUTBOUND_API_URL", "").strip()
    api_token = os.getenv("RINGCX_API_TOKEN", "").strip()
    webhook_secret = os.getenv("RINGCX_WEBHOOK_SECRET", "").strip()
    webhook_token = os.getenv("RINGCX_WEBHOOK_TOKEN", "").strip()
    from_number = os.getenv("RINGCX_NUMBER", "").strip()

    return {
        "enabled": enabled,
        "configured": bool(outbound_url),
        "outbound_api_configured": bool(outbound_url),
        "outbound_auth_configured": bool(api_token),
        "webhook_auth_configured": bool(webhook_secret or webhook_token),
        "has_from_number": bool(from_number),
        "has_default_to": bool(os.getenv("RINGCX_OUTBOUND_DEFAULT_TO", "").strip()),
    }


def _ringcx_is_enabled() -> bool:
    return os.getenv("RINGCX_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def _append_query(url: str, extra_params: dict[str, str]) -> str:
    parsed = urlparse(url)
    existing = parse_qs(parsed.query, keep_blank_values=True)
    for key, value in extra_params.items():
        existing[key] = [value]
    query = urlencode(existing, doseq=True)
    return urlunparse(parsed._replace(query=query))


def _extract_call_id(params: dict[str, str]) -> str:
    candidate_keys = [
        "call_id",
        "callId",
        "CallSid",
        "session_id",
        "sessionId",
        "conversationId",
        "interactionId",
    ]
    for key in candidate_keys:
        value = (params.get(key) or "").strip()
        if value:
            return value
    return ""


def _build_ringcx_media_ws_url(request: Request, session_id: str, call_id: str) -> str:
    explicit_ws = os.getenv("RINGCX_MEDIA_WS_URL", "").strip()
    if explicit_ws:
        base = explicit_ws
    else:
        public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
        if public_base:
            parsed = urlparse(public_base)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            base = f"{ws_scheme}://{parsed.netloc}/ringcx/media"
        else:
            req_url = request.url
            ws_scheme = "wss" if req_url.scheme == "https" else "ws"
            base = f"{ws_scheme}://{req_url.netloc}/ringcx/media"

    return _append_query(base, {"session_id": session_id, "call_id": call_id})


def _build_ringcx_agent_ws_url(session_id: str) -> str:
    base = os.getenv("RINGCX_AGENT_WS_URL", "ws://127.0.0.1:8000/ws/voice").strip()
    return _append_query(base, {"session_id": session_id})


def _ulaw_8k_to_pcm16_16k(payload_b64: str) -> bytes:
    try:
        mulaw_bytes = base64.b64decode(payload_b64)
    except Exception:
        return b""

    pcm8 = audioop.ulaw2lin(mulaw_bytes, 2)
    if not pcm8:
        return b""

    samples_8k = np.frombuffer(pcm8, dtype=np.int16)
    if samples_8k.size == 0:
        return b""

    samples_16k = np.repeat(samples_8k, 2)
    return samples_16k.astype(np.int16).tobytes()


def _pcm16_16k_to_ulaw_8k(pcm16_16k: bytes) -> bytes:
    if not pcm16_16k:
        return b""

    samples_16k = np.frombuffer(pcm16_16k, dtype=np.int16)
    if samples_16k.size == 0:
        return b""

    samples_8k = samples_16k[::2]
    pcm8 = samples_8k.astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm8, 2)


def _is_valid_ringcx_signature(raw_body: bytes, signature_header: str, secret: str) -> bool:
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    expected = {digest, f"sha256={digest}"}
    return any(hmac.compare_digest(signature_header, value) for value in expected)


def _extract_auth_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-RingCX-Token", "").strip()


def _validate_ringcx_request(request: Request, raw_body: bytes) -> bool:
    secret = os.getenv("RINGCX_WEBHOOK_SECRET", "").strip()
    expected_token = os.getenv("RINGCX_WEBHOOK_TOKEN", "").strip()

    if not secret and not expected_token:
        print("[RINGCX] WARNING: no webhook auth configured - skipping request validation (dev mode).")
        return True

    if secret:
        signature = (
            request.headers.get("X-RingCX-Signature", "").strip()
            or request.headers.get("X-RC-Signature", "").strip()
        )
        if not signature:
            print("[RINGCX] Signature validation failed: missing signature header")
            return False
        if not _is_valid_ringcx_signature(raw_body, signature, secret):
            print("[RINGCX] Signature validation failed: digest mismatch")
            return False

    if expected_token:
        supplied = _extract_auth_token(request)
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            print("[RINGCX] Token validation failed")
            return False

    return True


async def _extract_request_params(request: Request) -> tuple[dict[str, str], bytes]:
    if request.method.upper() == "GET":
        return dict(request.query_params), b""

    raw = await request.body()
    params: dict[str, str] = {}

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            if isinstance(data, dict):
                params = {str(k): str(v) for k, v in data.items()}
        except Exception:
            params = {}
    else:
        parsed = parse_qs(raw.decode("utf-8", errors="ignore"), keep_blank_values=True)
        params = {k: (v[0] if v else "") for k, v in parsed.items()}

    for key, value in request.query_params.items():
        params.setdefault(key, value)

    return params, raw


async def place_ringcx_outbound_call(payload: dict | None = None) -> tuple[dict, int]:
    if not _ringcx_is_enabled():
        return {"error": "RingCX provider is disabled (RINGCX_ENABLED=false)"}, 503

    body = payload or {}

    outbound_url = os.getenv("RINGCX_OUTBOUND_API_URL", "").strip()
    if not outbound_url:
        return {
            "error": "RINGCX_OUTBOUND_API_URL must be set in .env to place RingCX outbound calls"
        }, 500

    to_number = body.get("to") or os.getenv("RINGCX_OUTBOUND_DEFAULT_TO", "").strip()
    from_number = body.get("from") or os.getenv("RINGCX_NUMBER", "").strip()
    if not to_number:
        return {
            "error": "Provide 'to' in request body or set RINGCX_OUTBOUND_DEFAULT_TO in .env"
        }, 400

    to_field = os.getenv("RINGCX_OUTBOUND_TO_FIELD", "to").strip() or "to"
    from_field = os.getenv("RINGCX_OUTBOUND_FROM_FIELD", "from").strip() or "from"
    webhook_field = os.getenv("RINGCX_OUTBOUND_WEBHOOK_FIELD", "webhook_url").strip() or "webhook_url"
    status_field = os.getenv("RINGCX_OUTBOUND_STATUS_FIELD", "status_callback_url").strip() or "status_callback_url"

    public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    webhook_url = f"{public_base}/ringcx/voice/inbound" if public_base else ""
    status_callback_url = f"{public_base}/ringcx/voice/status" if public_base else ""

    outbound_payload: dict[str, object] = {
        to_field: to_number,
        webhook_field: webhook_url,
        status_field: status_callback_url,
    }
    if from_number:
        outbound_payload[from_field] = from_number

    provider_payload = body.get("provider_payload")
    if isinstance(provider_payload, dict):
        outbound_payload.update(provider_payload)

    extra_json = os.getenv("RINGCX_OUTBOUND_EXTRA_JSON", "").strip()
    if extra_json:
        try:
            extra_data = json.loads(extra_json)
            if isinstance(extra_data, dict):
                outbound_payload.update(extra_data)
        except Exception:
            print("[RINGCX] WARNING: invalid RINGCX_OUTBOUND_EXTRA_JSON - ignoring")

    headers = {"Content-Type": "application/json"}
    token = os.getenv("RINGCX_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    extra_headers_json = os.getenv("RINGCX_OUTBOUND_HEADERS_JSON", "").strip()
    if extra_headers_json:
        try:
            extra_headers = json.loads(extra_headers_json)
            if isinstance(extra_headers, dict):
                for key, value in extra_headers.items():
                    if key and value is not None:
                        headers[str(key)] = str(value)
        except Exception:
            print("[RINGCX] WARNING: invalid RINGCX_OUTBOUND_HEADERS_JSON - ignoring")

    timeout_seconds = float(os.getenv("RINGCX_OUTBOUND_TIMEOUT_SECONDS", "20").strip() or "20")

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(outbound_url, json=outbound_payload, headers=headers)
    except Exception as exc:
        print(f"[RINGCX] Outbound call failed: {exc}")
        return {"error": str(exc), "provider": "ringcx"}, 502

    response_text = response.text or ""
    parsed: dict | str
    try:
        parsed_json = response.json()
        parsed = parsed_json if isinstance(parsed_json, dict) else {"response": parsed_json}
    except Exception:
        parsed = response_text

    if response.status_code >= 400:
        return {
            "error": "RingCX outbound API rejected request",
            "provider": "ringcx",
            "provider_status": response.status_code,
            "provider_response": parsed,
        }, 502

    call_id = ""
    if isinstance(parsed, dict):
        call_id = (
            str(parsed.get("call_id") or "").strip()
            or str(parsed.get("callId") or "").strip()
            or str(parsed.get("id") or "").strip()
        )

    print(f"[RINGCX] Outbound call accepted to={to_number} from={from_number} call_id={call_id or 'unknown'}")
    return {
        "ok": True,
        "provider": "ringcx",
        "call_id": call_id,
        "to": to_number,
        "from": from_number,
        "provider_response": parsed,
    }, 200


@ringcx_router.api_route("/voice/inbound", methods=["GET", "POST"])
async def ringcx_voice_inbound(request: Request):
    if not _ringcx_is_enabled():
        return JSONResponse(content={"error": "RingCX provider disabled"}, status_code=503)

    params, raw_body = await _extract_request_params(request)

    if not _validate_ringcx_request(request, raw_body):
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)

    call_id = _extract_call_id(params)
    session_id = call_id or str(uuid.uuid4())
    media_ws_url = _build_ringcx_media_ws_url(request, session_id=session_id, call_id=call_id)

    print(f"[RINGCX] Inbound call mapped call_id={call_id or 'unknown'} session_id={session_id}")
    return JSONResponse(
        content={
            "ok": True,
            "provider": "ringcx",
            "session_id": session_id,
            "call_id": call_id,
            "media_ws_url": media_ws_url,
        },
        status_code=200,
    )


@ringcx_router.api_route("/voice/status", methods=["GET", "POST"])
async def ringcx_voice_status(request: Request):
    if not _ringcx_is_enabled():
        return JSONResponse(content={"error": "RingCX provider disabled"}, status_code=503)

    params, raw_body = await _extract_request_params(request)

    if not _validate_ringcx_request(request, raw_body):
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)

    call_id = _extract_call_id(params)
    status = (
        (params.get("status") or "").strip().lower()
        or (params.get("call_status") or "").strip().lower()
        or (params.get("CallStatus") or "").strip().lower()
    )

    print(f"[RINGCX] Status callback call_id={call_id or 'unknown'} status={status or 'unknown'}")

    if status in _TERMINAL_RINGCX_STATUSES and _session_store is not None:
        session = _session_store.pop(call_id, None)
        if session is not None:
            print(f"[RINGCX] Removed session from store for completed call call_id={call_id}")

    return JSONResponse(content={"ok": True, "provider": "ringcx"}, status_code=200)


@ringcx_router.post("/call/outbound")
async def ringcx_outbound_call(request: Request):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    result, status_code = await place_ringcx_outbound_call(body)
    return JSONResponse(content=result, status_code=status_code)


async def _forward_agent_audio_to_ringcx(agent_ws, ringcx_ws: WebSocket, stream_state: dict) -> None:
    last_generation = None

    try:
        while True:
            frame = await agent_ws.recv()
            if not isinstance(frame, (bytes, bytearray)):
                continue
            if len(frame) <= 4:
                continue

            stream_id = stream_state.get("stream_id")
            if not stream_id:
                continue

            generation = struct.unpack("<I", frame[:4])[0]
            pcm16_16k = bytes(frame[4:])
            ulaw_8k = _pcm16_16k_to_ulaw_8k(pcm16_16k)
            if not ulaw_8k:
                continue

            mode = stream_state.get("mode", "twilio_compatible")
            if last_generation is None:
                last_generation = generation
            elif generation != last_generation:
                if mode == "ringcx_json":
                    await ringcx_ws.send_text(json.dumps({"type": "clear", "streamId": stream_id}))
                else:
                    await ringcx_ws.send_text(json.dumps({"event": "clear", "streamSid": stream_id}))
                last_generation = generation

            payload = base64.b64encode(ulaw_8k).decode("ascii")
            if mode == "ringcx_json":
                outbound = {
                    "type": "media",
                    "streamId": stream_id,
                    "audio": {"payload": payload, "encoding": "mulaw", "sampleRate": 8000},
                }
            else:
                outbound = {
                    "event": "media",
                    "streamSid": stream_id,
                    "media": {"payload": payload},
                }
            await ringcx_ws.send_text(json.dumps(outbound))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[RINGCX] Agent->RingCX forward stopped: {exc}")


@ringcx_router.websocket("/media")
async def ringcx_media_bridge(websocket: WebSocket) -> None:
    if not _ringcx_is_enabled():
        await websocket.accept()
        await websocket.close(code=1013)
        return

    await websocket.accept()

    query = dict(websocket.query_params)
    session_id = query.get("session_id") or str(uuid.uuid4())
    call_id = query.get("call_id", "")
    mode = os.getenv("RINGCX_MEDIA_MODE", "twilio_compatible").strip().lower() or "twilio_compatible"

    agent_ws_url = _build_ringcx_agent_ws_url(session_id=session_id)
    stream_state = {"stream_id": "", "mode": mode}

    print(f"[RINGCX] Media bridge opening session_id={session_id} call_id={call_id or 'unknown'} mode={mode}")

    try:
        async with websockets.connect(
            agent_ws_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as agent_ws:
            forward_task = asyncio.create_task(_forward_agent_audio_to_ringcx(agent_ws, websocket, stream_state))

            try:
                while True:
                    raw = await websocket.receive_text()
                    event = json.loads(raw)
                    event_type = (event.get("event") or event.get("type") or "").strip().lower()

                    if event_type in {"start", "connected"}:
                        start = event.get("start", {}) if isinstance(event.get("start"), dict) else {}
                        stream_id = (
                            str(start.get("streamSid") or "").strip()
                            or str(start.get("streamId") or "").strip()
                            or str(event.get("streamSid") or "").strip()
                            or str(event.get("streamId") or "").strip()
                            or str(uuid.uuid4())
                        )
                        stream_state["stream_id"] = stream_id
                        call_id = (
                            str(start.get("callSid") or "").strip()
                            or str(start.get("callId") or "").strip()
                            or str(event.get("callSid") or "").strip()
                            or str(event.get("callId") or "").strip()
                            or call_id
                        )
                        print(
                            f"[RINGCX] Stream started stream_id={stream_id} "
                            f"call_id={call_id or 'unknown'} session_id={session_id}"
                        )

                    elif event_type == "media":
                        payload = (
                            str(event.get("media", {}).get("payload") or "").strip()
                            or str(event.get("audio", {}).get("payload") or "").strip()
                            or str(event.get("payload") or "").strip()
                        )
                        pcm16_16k = _ulaw_8k_to_pcm16_16k(payload)
                        if pcm16_16k:
                            await agent_ws.send(pcm16_16k)

                    elif event_type in {"stop", "end", "disconnect"}:
                        print(f"[RINGCX] Stream stop received session_id={session_id}")
                        break

                    elif event_type in {"dtmf", "mark", "heartbeat", "ping", "pong"}:
                        continue

            except WebSocketDisconnect:
                print(f"[RINGCX] Client disconnected session_id={session_id}")
            finally:
                if not forward_task.done():
                    forward_task.cancel()
                    try:
                        await forward_task
                    except asyncio.CancelledError:
                        pass

    except Exception as exc:
        print(f"[RINGCX] Media bridge error session_id={session_id}: {exc}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        print(f"[RINGCX] Media bridge closed session_id={session_id}")
