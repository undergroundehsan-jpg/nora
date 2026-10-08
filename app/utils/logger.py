import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.session import Session


_BASE_DIR = Path(__file__).resolve().parents[2]
_LOG_DIR = _BASE_DIR / "logs"
_TURN_LOG_PATH = _LOG_DIR / "call_transcripts.jsonl"
_SESSION_LOG_PATH = _LOG_DIR / "call_sessions.jsonl"
# Compliance / follow-up records written by the policy engine:
# suppression (do-not-call, wrong number, attorney, ...), callback requests,
# contact preferences, and safety escalations.
_EVENT_LOG_PATH = _LOG_DIR / "call_events.jsonl"
# One row per outbound attempt, including calls that were never answered.
_ATTEMPTS_LOG_PATH = _LOG_DIR / "call_attempts.jsonl"


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """
    Append a single JSON object as one line to the given JSONL file.
    Uses a background thread so we do not block the event loop.
    """

    def _write() -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[LOGGER] Failed to write to {path}: {e}")


async def log_turn(
    session: "Session",
    user_text: Optional[str],
    ai_text: Optional[str],
    primary_intent: Optional[str],
    turn_index: int,
) -> None:
    """
    Log a single conversational turn (user + assistant) to JSONL.
    """
    payload: Dict[str, Any] = {
        "type": "turn",
        "session_id": session.session_id,
        "lead_id": getattr(session, "lead_id", None),
        "timestamp": _iso_now(),
        "turn_index": turn_index,
        "user_transcript": user_text or "",
        "llm_response": ai_text or "",
        "intent": primary_intent,
        "refusal_count": session.refusal_count,
        "meeting_ask_count": session.meeting_ask_count,
        "agreed_to_baa": session.agreed_to_baa,
        "agreed_to_meeting": session.agreed_to_meeting,
        "email_captured": session.email_captured,
        "email_address": session.email_address,
        "current_mode": session.current_mode,
        "stage": getattr(session, "call_stage", None),
    }
    await _append_jsonl(_TURN_LOG_PATH, payload)


async def log_turn_latency(session: "Session") -> None:
    """Log per-turn latency marks (ms since end of user speech), if any were recorded."""
    latency = session.turn_latency_ms()
    if not latency:
        return
    await _append_jsonl(_TURN_LOG_PATH, {
        "type": "latency",
        "session_id": session.session_id,
        "timestamp": _iso_now(),
        "turn_index": session.turn_count,
        "latency_ms": latency,
    })


async def log_call_attempt(record: Dict[str, Any]) -> None:
    """Log one Twilio call attempt (no session needed — it may never connect)."""
    await _append_jsonl(_ATTEMPTS_LOG_PATH, {"timestamp": _iso_now(), **record})


async def log_blocked_call(record: Dict[str, Any]) -> None:
    """Record a call that compliance rules stopped before it was dialled."""
    await _append_jsonl(_EVENT_LOG_PATH, {
        "timestamp": _iso_now(),
        "type": "blocked_call",
        **record,
    })


async def log_event(session: "Session", record: Dict[str, Any]) -> None:
    """Persist a policy record (suppression, callback request, preference, escalation)."""
    lead = session.lead_context or {}
    account_id = str(lead.get("member_id") or "")
    payload: Dict[str, Any] = {
        "session_id": session.session_id,
        "lead_id": getattr(session, "lead_id", None),
        "timestamp": _iso_now(),
        "lead_name": lead.get("lead_name"),
        # Masked: the full identifier is not needed downstream.
        "account_id": f"***{account_id[-4:]}" if account_id else None,
        "phone": lead.get("phone"),
        **record,
    }
    await _append_jsonl(_EVENT_LOG_PATH, payload)


def _infer_final_outcome(session: "Session") -> str:
    """
    Derive a coarse-grained final outcome label from the session state.
    This is intentionally simple so it can be refined later in Supabase.
    """
    disposition = getattr(session, "final_disposition", None)
    if disposition:
        return disposition
    outcome = getattr(session, "outcome", None)
    if outcome:
        return outcome
    if session.agreed_to_meeting:
        return "SUCCESS_MEETING"
    if session.agreed_to_baa or session.email_captured:
        return "SUCCESS_BAA_OR_EMAIL"
    if session.refusal_count >= 3:
        return "REJECTED_AFTER_3_REFUSALS"
    if session.last_intent in {"ALREADY_HAVE_BILLER", "ALREADY_AUDITED"}:
        return "REJECTED_ALREADY_HAVE_BILLER_OR_AUDITED"
    if session.last_intent == "NOT_INTERESTED":
        return "REJECTED_NOT_INTERESTED"
    return "UNKNOWN_OUTCOME"


async def log_session_summary(session: "Session") -> None:
    """
    Log a summary row for the entire call/session when it ends.
    """
    outcome = _infer_final_outcome(session)

    payload: Dict[str, Any] = {
        "type": "session",
        "session_id": session.session_id,
        "lead_id": getattr(session, "lead_id", None),
        "start_time": datetime.utcfromtimestamp(session.start_time).isoformat() + "Z",
        "end_time": _iso_now(),
        "total_turns": session.turn_count,
        "refusal_count": session.refusal_count,
        "meeting_ask_count": session.meeting_ask_count,
        "last_intent": session.last_intent,
        "last_mode": session.current_mode,
        "agreed_to_baa": session.agreed_to_baa,
        "agreed_to_meeting": session.agreed_to_meeting,
        "email_captured": session.email_captured,
        "email_address": session.email_address,
        "final_outcome": outcome,
        "identity_verified": getattr(session, "identity_verified", None),
        "reminder_delivered": getattr(session, "reminder_delivered", None),
        "callback_time": getattr(session, "callback_time", None),
        "escalation_reason": getattr(session, "escalation_reason", None),
        "preferences": getattr(session, "preferences", None),
        "mood_trajectory": getattr(session, "mood_trajectory", None),
    }
    await _append_jsonl(_SESSION_LOG_PATH, payload)

