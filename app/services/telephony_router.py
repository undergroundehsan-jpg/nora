import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.ringcx_bridge import get_ringcx_provider_status, place_ringcx_outbound_call
from app.services.twilio_bridge import (
    check_twilio_connectivity,
    get_twilio_provider_status,
    place_twilio_outbound_call,
)

telephony_router = APIRouter()


def _truthy_env(name: str, default: str = "") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"", "0", "false", "no"}


@telephony_router.get("/providers")
async def telephony_providers_status() -> dict:
    return {
        "ok": True,
        "default_provider": os.getenv("TELEPHONY_DEFAULT_PROVIDER", "twilio").strip().lower() or "twilio",
        "providers": {
            "twilio": get_twilio_provider_status(),
            "ringcx": get_ringcx_provider_status(),
        },
    }


@telephony_router.get("/twilio/ping")
async def validate_twilio_connectivity() -> dict:
    """Validate Twilio account auth/connectivity from this running environment."""
    return await check_twilio_connectivity()


@telephony_router.get("/ringcx/validate")
async def validate_ringcx_config() -> dict:
    public_base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    outbound_url = os.getenv("RINGCX_OUTBOUND_API_URL", "").strip()
    api_token = os.getenv("RINGCX_API_TOKEN", "").strip()
    webhook_secret = os.getenv("RINGCX_WEBHOOK_SECRET", "").strip()
    webhook_token = os.getenv("RINGCX_WEBHOOK_TOKEN", "").strip()
    from_number = os.getenv("RINGCX_NUMBER", "").strip()
    default_to = os.getenv("RINGCX_OUTBOUND_DEFAULT_TO", "").strip()
    ringcx_enabled = _truthy_env("RINGCX_ENABLED", "true")

    missing = []
    if not ringcx_enabled:
        missing.append("RINGCX_ENABLED=true")
    if not public_base:
        missing.append("PUBLIC_BASE_URL")
    if not outbound_url:
        missing.append("RINGCX_OUTBOUND_API_URL")
    if not api_token:
        missing.append("RINGCX_API_TOKEN")
    if not from_number:
        missing.append("RINGCX_NUMBER")
    if not default_to:
        missing.append("RINGCX_OUTBOUND_DEFAULT_TO")
    if not (webhook_secret or webhook_token):
        missing.append("RINGCX_WEBHOOK_SECRET or RINGCX_WEBHOOK_TOKEN")

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "webhooks": {
            "inbound": f"{public_base}/ringcx/voice/inbound" if public_base else "",
            "status": f"{public_base}/ringcx/voice/status" if public_base else "",
            "media": f"{public_base.replace('https://', 'wss://').replace('http://', 'ws://')}/ringcx/media"
            if public_base else "",
        },
        "outbound_fields": {
            "to": os.getenv("RINGCX_OUTBOUND_TO_FIELD", "to").strip() or "to",
            "from": os.getenv("RINGCX_OUTBOUND_FROM_FIELD", "from").strip() or "from",
            "webhook": os.getenv("RINGCX_OUTBOUND_WEBHOOK_FIELD", "webhook_url").strip() or "webhook_url",
            "status": os.getenv("RINGCX_OUTBOUND_STATUS_FIELD", "status_callback_url").strip() or "status_callback_url",
        },
    }


@telephony_router.post("/call/outbound")
async def telephony_outbound_call(request: Request):
    """
    Unified outbound endpoint.

    JSON body:
      {
        "provider": "twilio|ringcx",
        "to": "+923...",
        "from": "+1478...",
        "provider_payload": {}
      }
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    provider = (
        str(body.get("provider") or os.getenv("TELEPHONY_DEFAULT_PROVIDER", "twilio"))
        .strip()
        .lower()
    )

    if provider == "twilio":
        result, status_code = await place_twilio_outbound_call(body)
        return JSONResponse(content=result, status_code=status_code)

    if provider == "ringcx":
        result, status_code = await place_ringcx_outbound_call(body)
        return JSONResponse(content=result, status_code=status_code)

    return JSONResponse(
        content={
            "error": "Unsupported provider",
            "supported_providers": ["twilio", "ringcx"],
        },
        status_code=400,
    )
