import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time
import struct
import json
import re
from typing import Dict

from app.session import Session, AssistantState, SAMPLE_LEAD_CONTEXT
from app.services.audio_vad import vad_service, _vad_executor
from app.services.stt import DeepgramSTTStream
from app.services.llm import get_ai_response_stream, warm_up_llm_connection, REPLY_MODEL
from app.services.tts import (
    process_tts_queue,
    close_tts_connection,
    play_cached_audio,
    speak_text,
    warm_up_tts_connection,
)
from app.services.intent import (
    classify_intent,
    OutreachIntent,
    ContentTier,
    CLASSIFIER_MODEL,
    warm_up_classifier_connection,
)
from app.services.intent import _heuristic_intent
from app.services import policy
from app.services.leads import LeadError, lead_for_call


def heuristic_intent_for(transcript: str) -> OutreachIntent:
    return _heuristic_intent(transcript)[0]
from app.services.twilio_bridge import (
    twilio_router,
    register_session_store,
    get_twilio_provider_status,
)
from app.services.ringcx_bridge import (
    ringcx_router,
    register_ringcx_session_store,
    get_ringcx_provider_status,
)
from app.services.telephony_router import telephony_router
from app.api.dashboard import dashboard_router
from app.utils.language import (
    detect_language,
    get_thinking_filler,
    get_speculative_filler,
    build_greeting_variants,
)
from app.utils.logger import log_turn, log_session_summary, log_turn_latency, log_event
import traceback
import random
import os

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await _check_models_available()
    yield


app = FastAPI(title="Voice Assistant Backend", lifespan=_lifespan)
app.include_router(twilio_router, prefix="/twilio", tags=["twilio"])
app.include_router(ringcx_router, prefix="/ringcx", tags=["ringcx"])
app.include_router(telephony_router, prefix="/telephony", tags=["telephony"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])

# CORS middleware configuration for frontend + ngrok + Netlify deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Analytics dashboard (separate Next app, so it runs on another port)
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://early-guiding-feline.ngrok-free.app",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve compiled frontend (if present) ──────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]
_frontend_candidates = [
    _BASE_DIR / "frontend" / "out",     # next export output (preferred)
    _BASE_DIR / "frontend" / "public",  # static assets with index.html
    _BASE_DIR / "frontend",
]
for _candidate in _frontend_candidates:
    if _candidate.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(_candidate), html=True),
            name="frontend",
        )
        print(f"[Frontend] Serving static files from: {_candidate}")
        break

# Global store for session persistence across disconnects
# Keys are session_id strings, values are Session objects.
# We don't delete them immediately on disconnect so users can resume.
session_store: Dict[str, Session] = {}

# Give the Twilio status callback a reference so it can evict terminated sessions.
register_session_store(session_store)
register_ringcx_session_store(session_store)

# Strong references for background tasks to prevent GC from destroying them mid-execution
_background_tasks = set()

# Pause after a closing line before closing the socket. Closing helpers already
# wait for the estimated playback end (+0.5 s), so this only covers transport lag.
try:
    END_CALL_GRACE_SECS = max(0.0, float(os.getenv("END_CALL_GRACE_SECS", "0.5")))
except ValueError:
    END_CALL_GRACE_SECS = 0.5

# Filled in at startup: which configured models the API key can actually use.
_model_status: Dict[str, object] = {"checked": False}


async def _check_models_available():
    """Fail loudly (in logs and /health) when a configured Groq model is gone."""
    from app.services.intent import _client as groq_client
    try:
        available = {m.id for m in (await groq_client.models.list()).data}
    except Exception as e:
        print(f"[STARTUP] Could not list Groq models: {e}")
        _model_status.update({"checked": False, "error": str(e)})
        return
    missing = [m for m in (REPLY_MODEL, CLASSIFIER_MODEL) if m not in available]
    _model_status.update({
        "checked": True,
        "reply_model": REPLY_MODEL,
        "classifier_model": CLASSIFIER_MODEL,
        "missing": missing,
    })
    if missing:
        print(
            f"[STARTUP] ERROR: Groq model(s) unavailable: {', '.join(missing)}. "
            f"Set GROQ_REPLY_MODEL / GROQ_CLASSIFIER_MODEL in .env. "
            f"Available chat models: {', '.join(sorted(m for m in available if 'whisper' not in m and 'guard' not in m))}"
        )
    else:
        print(f"[STARTUP] Groq models OK (reply={REPLY_MODEL}, classifier={CLASSIFIER_MODEL})")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models": _model_status,
        "default_provider": os.getenv("TELEPHONY_DEFAULT_PROVIDER", "twilio").strip().lower() or "twilio",
        "providers": {
            "twilio": get_twilio_provider_status(),
            "ringcx": get_ringcx_provider_status(),
        },
    }


def _predict_speculative_intent_and_tier(transcript: str) -> tuple[str, str]:
    """Cheap local guess used to start the LLM before classification finishes.

    Only intents the LLM phrases are predicted. Anything the policy engine
    answers with a fixed line (refusals, callbacks, compliance, escalations)
    returns UNCLEAR so no speculative speech starts for it. Guesses must match
    classify_intent's heuristic labels, otherwise the speculative run is
    discarded and restarted.
    """
    t = (transcript or "").strip().lower()
    if not t:
        return "UNCLEAR", "COMMAND"

    # Anything that looks policy-handled: don't speculate.
    if re.search(
        r"\b(call|stop|remove|wrong|lawyer|attorney|bankrupt|passed away|paid|owe|afford|plan|"
        r"installment|pay|reschedule|later|tomorrow|busy|not interested|no thanks|hold on|one sec|"
        r"repeat|slow|bye|person|human|agent|robot|bot|ai)\b",
        t,
    ):
        return "UNCLEAR", "COMMAND"

    if re.search(r"\b(email|e-mail|mail me|in writing)\b", t):
        return "EMAIL_REQUEST", "COMMAND"
    if re.search(r"\b(scam|spam|fraud|legit|don't trust)\b", t):
        return "TRUST_CONCERN", "OBJECTION"
    if re.search(r"\b(who is this|who's this|what is this about|what's this regarding|what is this regarding)\b", t):
        return "GATEKEEPER", "COMMAND"

    return "UNCLEAR", "COMMAND"


_TELEPHONY_PROMPT_PATTERNS = [
    r"\bhold while i try to connect you\b",
    r"\bplease hold\b",
    r"\bcall has been forwarded\b",
    r"\bparty you are trying to reach\b",
    r"\bcurrently unavailable\b",
    r"\bplease leave (?:a )?message\b",
    r"\bafter the tone\b",
    r"\bvoicemail\b",
    r"\bmailbox\b",
    r"\bnumber you have dialed\b",
    r"\byour call (?:cannot|can not) be completed\b",
]

_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _looks_like_telephony_prompt(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False
    return any(re.search(pattern, t) for pattern in _TELEPHONY_PROMPT_PATTERNS)


def _looks_like_partial_date_fragment(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    tokens = t.split()
    if not any(token in _MONTH_WORDS for token in tokens):
        return False

    if re.search(r"\b(19|20)\d{2}\b", t):
        return False

    if re.search(r"\b(two|twenty)$", t):
        return True
    if re.search(r"\btwo thousand(?: twenty)?$", t):
        return True

    return "effective date" in t and tokens[-1] in {"two", "twenty"}


def _get_last_assistant_message(session: Session) -> str:
    for msg in reversed(session.chat_history):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def _last_assistant_asked_question(session: Session) -> bool:
    last = _get_last_assistant_message(session).strip()
    if not last:
        return False
    return "?" in last


def _last_assistant_was_benefit_question(session: Session) -> bool:
    last = _get_last_assistant_message(session).lower()
    if not last:
        return False
    return bool(re.search(
        r"(copay|coinsurance|deductible|out[- ]?of[- ]?pocket|oop|prior auth|authorization|pre-?auth|"
        r"referral|visit limits?|cpt|covered|claims (mailing )?address|payer id|call reference|in[- ]?network|"
        r"out[- ]?of[- ]?network|effective date|plan type)",
        last,
    ))


def _looks_like_short_benefit_answer(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    if _looks_like_short_confirmation_answer(t):
        return True

    t_words = re.sub(r"[^a-z0-9%/ @.-]", "", t)
    t_clean = t_words.strip(" .!? ,")
    tokens = [token for token in t_clean.split() if token]

    yes_no_tokens = {"yes", "yeah", "yep", "no", "nope", "nah", "ok", "okay", "correct", "right"}
    if tokens and all(token in yes_no_tokens for token in tokens):
        return True

    if re.search(r"\b\d{1,3}%\b", t_clean):
        return True

    if re.fullmatch(r"(yes|yeah|yep|no|nope|nah|correct|right|ok|okay)", t_clean):
        return True
    if re.fullmatch(r"(we )?(do not|don't) need that", t_clean):
        return True
    if re.fullmatch(r"(not required|not needed)", t_clean):
        return True

    if re.search(r"\d", t_clean) and len(tokens) <= 4:
        return True
    if re.fullmatch(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?", t_clean):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", t_clean):
        return True
    if any(month in t_clean for month in _MONTH_WORDS) and re.search(r"\b(19|20)\d{2}\b", t_clean):
        return True

    if re.search(r"\bcovered\b", t_clean):
        return True
    if re.search(r"\b(eligible|active)\b", t_clean):
        return True
    if re.search(r"\b(in|out)[- ]?network\b", t_clean):
        return True
    if re.search(r"\b(ppo|hmo|epo|pos|hsa|hdhp)\b", t_clean):
        return True
    if re.search(r"\b(no|not) (referral|prior auth|authorization|pre-?auth)\b", t_clean):
        return True
    if re.search(r"\b[A-Z]{2,5}\s?\d{4,8}\b", t_clean, flags=re.IGNORECASE):
        return True

    return False


def _looks_like_short_confirmation_answer(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    t_clean = re.sub(r"[^a-z0-9 ]", "", t).strip()
    if not t_clean:
        return False

    patterns = [
        r"^(yes|yeah|yep|no|nope|nah) (it|that|this) (is|isnt|is not|was|wasnt|was not)$",
        r"^(it|that|this) (is|isnt|is not|was|wasnt|was not)$",
        r"^(yes|yeah|yep|no|nope|nah) thats (correct|right)$",
        r"^(thats|that is) (correct|right)$",
    ]
    return any(re.fullmatch(p, t_clean) for p in patterns)


def _looks_like_short_identifier_answer(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    t = re.sub(r"[^a-z0-9 ]", "", t)

    return bool(re.fullmatch(
        r"(member id|member number|dob|date of birth|patient name|name|npi|tax id|tin)",
        t,
    ))


def _looks_like_short_contact_answer(transcript: str) -> bool:
    t = (transcript or "").strip()
    if not t:
        return False

    return bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t))


def _looks_like_incomplete_benefit_answer(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    if _looks_like_short_confirmation_answer(t):
        return False

    t_clean = re.sub(r"[^a-z0-9 ]", "", t).strip()
    tokens = t_clean.split()
    if not tokens:
        return False

    trailing_stopwords = {
        "the", "a", "an", "for", "to", "of", "and", "or", "but", "so",
        "is", "are", "was", "were", "this", "that", "these", "those",
    }
    if len(tokens) <= 4 and tokens[-1] in trailing_stopwords:
        return True

    if "effective date" in t_clean and not re.search(r"\b(19|20)\d{2}\b", t_clean):
        return True

    if any(month in t_clean for month in _MONTH_WORDS) and not re.search(r"\b(19|20)\d{2}\b", t_clean):
        if len(tokens) <= 2:
            return True

    incomplete_patterns = [
        r"^(this is)$",
        r"^(this is the)$",
        r"^(the|the right|the right department)$",
        r"^(the )?(effective date|plan type|deductible|copay|coinsurance|out of pocket|oop)( is)?$",
        r"^(the )?(member id|payer id|claims address)( is)?$",
    ]
    return any(re.match(p, t_clean) for p in incomplete_patterns)


def _looks_like_clear_refusal(transcript: str) -> bool:
    t = (transcript or "").lower()
    return bool(re.search(
        r"\b(not interested|do not call|don't call|stop calling|wrong number|no thanks|no thank you)\b",
        t,
    ))


def _looks_like_benefit_response(transcript: str) -> bool:
    t = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not t:
        return False

    benefit_patterns = [
        r"\beligible\b", r"\bactive\b", r"\bcovered\b",
        r"\bcopay\b", r"\bcoinsurance\b", r"\bdeductible\b",
        r"\bout[- ]?of[- ]?pocket\b", r"\boop\b",
        r"\bprior auth(?:orization)?\b", r"\bpre[- ]?auth(?:orization)?\b",
        r"\breferral\b", r"\bvisit limits?\b",
        r"\bin[- ]?network\b", r"\bout[- ]?of[- ]?network\b",
        r"\bplan type\b", r"\beffective date\b",
        r"\bclaims (mailing )?address\b", r"\bpayer id\b",
        r"\bcall reference\b",
        r"\balready have it\b",
        r"\byou already have it\b",
    ]

    return any(re.search(p, t) for p in benefit_patterns)

async def _report_turn_latency(session: Session) -> None:
    """Print and log this turn's latency marks. Never raises."""
    try:
        latency = session.turn_latency_ms()
        if latency:
            ordered = ", ".join(f"{k}={v}ms" for k, v in sorted(latency.items(), key=lambda kv: kv[1]))
            print(f"[Session {session.session_id}] [LATENCY] turn {session.turn_count}: {ordered}")
            await log_turn_latency(session)
    except Exception:
        traceback.print_exc()


async def run_llm_pipeline(
    transcript: str,
    lang: str,
    session: Session,
    ws_audio_queue: asyncio.Queue,
    max_tokens: int = 100,
    max_words: int = 40,
    tier: str = "COMMAND",
    primary_intent: str = "UNCLEAR",
    filler_text: str = "",
    filler_used: bool = False,
    current_mode: str = "INTRO",
    pitch_delivered: bool = False,
):
    """
    Runs the LLM and TTS pipeline for a single user turn.
    max_tokens/max_words come from the intent classifier.
    If filler_text is set, it is pushed to the TTS queue FIRST so it plays
    immediately while the LLM is still generating — masking startup latency.
    """
    tts_queue = asyncio.Queue()
    stage_at_start = session.call_stage

    # If we have a thinking filler, push it first so it starts playing
    # while the LLM is still generating. This masks LLM startup latency.
    if filler_text:
        await tts_queue.put(filler_text)

    # Spawn TTS processor for this turn (it exits when it receives None)
    tts_task = asyncio.create_task(process_tts_queue(
        session,
        tts_queue,
        ws_audio_queue,
        primary_intent=primary_intent,
        tier=tier,
        refusal_count=session.refusal_count,
    ))
    cancelled = False

    try:
        response = await get_ai_response_stream(
            transcript, lang, session, tts_queue,
            max_tokens=max_tokens,
            max_words=max_words,
            tier=tier,
            primary_intent=primary_intent,
            filler_used=filler_used,
            current_mode=current_mode,
            recent_openers=getattr(session, 'recent_openers', []),
            pitch_delivered=pitch_delivered,
        )

        if response and not session.interrupt_llm.is_set():
            session.last_response = response
            policy.note_assistant_turn(session, stage_at_start, response)
            # Track opening words for variety across turns
            opener_words = response.strip().split()[:3]
            if opener_words:
                if not hasattr(session, 'recent_openers'):
                    session.recent_openers = []
                session.recent_openers.append(" ".join(opener_words))
                if len(session.recent_openers) > 5:
                    session.recent_openers = session.recent_openers[-5:]
            # Fire-and-forget local JSONL turn logging. Errors are non-fatal.
            try:
                await log_turn(
                    session=session,
                    user_text=session.last_user_input or transcript,
                    ai_text=response,
                    primary_intent=session.last_intent,
                    turn_index=session.turn_count,
                )
            except Exception:
                traceback.print_exc()
    except asyncio.CancelledError:
        cancelled = True
        session.interrupt_llm.set()
        session.interrupt_tts.set()
        raise
    finally:
        # Ensure TTS worker never hangs waiting for a sentinel when this pipeline is cancelled.
        try:
            await tts_queue.put(None)
        except Exception:
            pass

        if not tts_task.done():
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

        # Restore state to LISTENING if we weren't interrupted.
        current_state = await session.get_state()
        if current_state in {AssistantState.SPEAKING, AssistantState.THINKING}:
            await session.set_state(AssistantState.LISTENING)

        # Free up the reference since we're done.
        if session.llm_task == asyncio.current_task():
            session.llm_task = None

        if cancelled:
            print(f"[Session {session.session_id}] [LLM] Pipeline cancelled cleanly.")
        else:
            await _report_turn_latency(session)



@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None, lead_id: str = None):
    """
    WebSocket endpoint for handling full-duplex voice communication.
    Accepts an optional session_id query param to resume previous chat history.
    - Receives 16kHz PCM audio from the client.
    - Sends 16kHz PCM audio TTS back to the client.
    """
    await websocket.accept()
    
    # Check if this is a reconnection attempt
    is_reconnect = False
    if session_id and session_id in session_store:
        session = session_store[session_id]
        session.websocket = websocket # Update to the new connection
        is_reconnect = True
        print(f"[Session {session_id}] Reconnected! Resuming history ({len(session.chat_history)} messages).")
    else:
        # Brand new session
        session_id = session_id or str(uuid.uuid4())
        session = Session(websocket=websocket, session_id=session_id)
        # The customer for this call. Without a lead_id this falls back to the
        # sample customer, which ALLOW_SAMPLE_LEAD=false disables for production.
        try:
            session.lead_context = lead_for_call(lead_id or "")
        except LeadError as e:
            print(f"[Session {session_id}] [LEADS] {e} — refusing the call.")
            await websocket.close(code=1008)
            return
        session.lead_id = session.lead_context.get("lead_id")
        print(f"[Session {session_id}] [LEADS] Customer: {session.lead_id} "
              f"({session.lead_context.get('lead_name', 'unknown')})")
        session_store[session_id] = session
        print(f"[Session {session_id}] Connected. Starting new session.")
    
    # ── Start Deepgram streaming STT (one connection per session) ──
    # Reuse existing stream across reconnects and turns; only reconnect if dropped.
    if session.stt_stream is None:
        session.stt_stream = DeepgramSTTStream()
    # Don't await STT connection here — launch it in background so it
    # warms up while the greeting plays. This shaves ~1-2s off startup.
    stt_connect_task = asyncio.create_task(session.stt_stream.ensure_connection())
    
    ws_audio_queue = asyncio.Queue()
    
    # Background task to send audio chunks back to the browser
    async def send_audio_task():
        try:
            while True:
                chunk = await ws_audio_queue.get()
                if chunk is None:
                    continue
                # Prepend 4-byte little-endian generation header
                gen = session.tts_generation
                header = struct.pack('<I', gen)
                await websocket.send_bytes(header + chunk)
                session.audio_chunks_sent += 1
                if "speech_end" in session.turn_metrics:
                    session.mark_turn("first_audio_sent")
        except (asyncio.CancelledError, WebSocketDisconnect):
            # BUG 3 FIX: explicitly handle WebSocketDisconnect so it doesn't
            # get swallowed by the bare Exception handler below and hide the error.
            pass
        except Exception as e:
            print(f"[Session {session_id}] Sender error: {e}")

    sender_task = asyncio.create_task(send_audio_task())

    async def _force_stop_outbound_audio(reason: str):
        """Invalidate and clear pending outbound audio so forced wrap-up is immediate."""
        session.interrupt_llm.set()
        session.interrupt_tts.set()
        session.tts_generation += 1
        session.expected_speech_end_time = time.time()

        drained = 0
        while True:
            try:
                ws_audio_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        if drained:
            print(f"[Session {session_id}] [{reason}] Cleared {drained} stale outbound chunk(s).")

        try:
            await websocket.send_text(json.dumps({
                "type": "CLEAR",
                "generation": session.tts_generation,
            }))
        except Exception:
            pass

    async def _cancel_llm_task():
        if session.llm_task and not session.llm_task.done():
            session.llm_task.cancel()
            try:
                await session.llm_task
            except asyncio.CancelledError:
                pass
        session.llm_task = None

    async def _persist_policy_records(action: "policy.PolicyAction"):
        for record in action.records:
            try:
                await log_event(session, record)
            except Exception:
                traceback.print_exc()

    def _remember_exchange(user_text: str, assistant_text: str):
        if user_text:
            session.chat_history.append({"role": "user", "content": user_text})
        session.chat_history.append({"role": "assistant", "content": assistant_text})
        if len(session.chat_history) > 16:
            session.chat_history[:] = session.chat_history[-16:]

    async def _ensure_goodbye_played(chunks_before: int, cache_name: str = "decline_bye"):
        """If nothing reached the caller since `chunks_before`, play a cached goodbye."""
        await asyncio.sleep(0.05)  # let the sender flush anything still queued
        if session.audio_chunks_sent > chunks_before:
            return
        print(f"[Session {session_id}] [GOODBYE] Live TTS produced no audio; playing cached '{cache_name}'.")
        session.clear_interrupts()
        await play_cached_audio(session, cache_name, ws_audio_queue, fallback_text="")

    async def _speak_fixed_line(text: str, cache_key: str = None):
        """Speak a fixed line, from pre-recorded audio when one matches the text."""
        if cache_key:
            await play_cached_audio(
                session, f"tpl_{cache_key}", ws_audio_queue,
                fallback_text=text, expect_text=text,
            )
        else:
            await speak_text(session, text, ws_audio_queue)

    async def _end_call_with_line(text: str, disposition: str, reason: str, user_text: str = "",
                                  cache_key: str = None):
        """Deterministic close: stop stale audio, speak one line, hang up."""
        print(f"[Session {session_id}] [POLICY] Ending call ({reason}) disposition={disposition}")
        await _force_stop_outbound_audio(reason)
        await _cancel_llm_task()
        session.clear_interrupts()
        if disposition and not session.final_disposition:
            session.final_disposition = disposition
        _remember_exchange(user_text, text)
        session.last_response = text
        try:
            await log_turn(session=session, user_text=user_text, ai_text=text,
                           primary_intent=reason, turn_index=session.turn_count)
        except Exception:
            traceback.print_exc()
        # These helpers wait until the audio has physically played out.
        chunks_before = session.audio_chunks_sent
        await _speak_fixed_line(text, cache_key)
        await _ensure_goodbye_played(chunks_before)
        await _report_turn_latency(session)
        await asyncio.sleep(END_CALL_GRACE_SECS)
        await websocket.close(code=1000)

    async def _say_line(text: str, reason: str, user_text: str = "", cache_key: str = None):
        """Speak a fixed line in the background and keep listening (barge-in stays live)."""
        print(f"[Session {session_id}] [POLICY] {reason}: {text}")
        _remember_exchange(user_text, text)
        if reason not in {"REPEAT_REQUEST", "SLOW_DOWN"}:
            session.last_response = text
        try:
            await log_turn(session=session, user_text=user_text, ai_text=text,
                           primary_intent=reason, turn_index=session.turn_count)
        except Exception:
            traceback.print_exc()

        async def _speak():
            try:
                await _speak_fixed_line(text, cache_key)
            finally:
                # If TTS produced no audio the state never reached SPEAKING;
                # make sure the turn doesn't stay stuck in THINKING.
                if await session.get_state() in {AssistantState.SPEAKING, AssistantState.THINKING}:
                    await session.set_state(AssistantState.LISTENING)
            await _report_turn_latency(session)
            if session.llm_task == asyncio.current_task():
                session.llm_task = None

        session.llm_task = asyncio.create_task(_speak())

    audio_chunk_count = 0          # Count of audio chunks received while listening
    vad_buffer = []                 # Accumulates exact 512 samples for VAD
    vad_samples_accumulated = 0     # Count of samples in vad_buffer
    silence_frames = 0              # Used for silence detection
    
    # ~190ms of silence (6 frames × 32ms per 512-sample VAD chunk @ 16kHz).
    # Loosened to avoid cutting off mid-sentence pauses.
    # The audio_chunk_count > 10 guard and Deepgram endpointing fallback
    # prevent false positives from any premature cutoff.
    SILENCE_THRESHOLD_FRAMES = 6
    
    # ~1.1 seconds. If the user makes a brief noise (like coughing) but doesn't
    # say a full sentence, wipe the buffer so it doesn't accumulate forever.
    WIPE_BUFFER_SILENCE_FRAMES = 35
    
    last_active_time = time.time()
    session.last_active_time = last_active_time
    
    greeting_text = ""
    greeting_cache_name = ""
    greeting_replayed_once = False

    # 1. Send Auto-Greeting ONLY if this is a new session
    if not is_reconnect:
        # Build a personalized greeting using lead context if available
        lead_name = ""
        practice_name = "the practice"
        
        if session.lead_context:
            lead_name = session.lead_context.get("lead_name", "")
            practice_name = session.lead_context.get("practice_name", "the practice")

        variants = build_greeting_variants(lead_name, practice_name)
        variant_idx = random.randrange(len(variants))
        cache_name, greeting = variants[variant_idx]

        greeting_text = greeting
        greeting_cache_name = cache_name

        # Append greeting to chat history so the LLM has context for the user's first response
        session.chat_history.append({"role": "assistant", "content": greeting})

        print(f"[Session {session_id}] [GREETING] Sending greeting (cached, interruptible)...")
        # Play from pre-recorded cache (~0ms) instead of live TTS API (~1-2s).
        # Falls back to live TTS if the .pcm file doesn't exist yet.
        asyncio.create_task(play_cached_audio(
            session, cache_name, ws_audio_queue,
            fallback_text=greeting, expect_text=greeting,
        ))

        # Warm up Groq connection pool while the greeting plays.
        # The first real Groq call (intent classification) suffers an extra
        # ~500-800ms TCP+TLS cold-start.  This throwaway 1-token call pays
        # that cost in the background so Turn 1 intent latency is ~200-400ms
        # instead of >1000ms.
        asyncio.create_task(warm_up_classifier_connection())
        # The response LLM and Rime use separate connection pools; open them too.
        asyncio.create_task(warm_up_llm_connection())
        asyncio.create_task(warm_up_tts_connection())
    else:
        # Optional: send a brief reconnection acknowledgment
        asyncio.create_task(play_cached_audio(
            session, "reconnected", ws_audio_queue,
            fallback_text="Reconnected.",
        ))

    # Deepgram connects in the background (stt_connect_task) while greeting plays.
    # The while loop starts immediately so VAD interruption detection works during greeting.

    inactivity_timeout_s = 45.0
    try:
        inactivity_timeout_s = max(10.0, float(os.getenv("CALL_INACTIVITY_TIMEOUT_SECS", "45")))
    except ValueError:
        inactivity_timeout_s = 45.0

    try:
        while True:
            # 0. Read state once per iteration and reuse — avoids multiple
            # async lock acquisitions per audio chunk in the hot path.
            current_state = await session.get_state()

            if session.close_after_speaking and current_state == AssistantState.LISTENING:
                if session.llm_task is None or session.llm_task.done():
                    print(f"[Session {session_id}] Closing after scripted goodbye.")
                    session.final_disposition = session.final_disposition or session.outcome or "COMPLETED"
                    await asyncio.sleep(END_CALL_GRACE_SECS)
                    await websocket.close(code=1000)
                    break

            # The response LLM failed twice in a row: say goodbye instead of going silent.
            if (
                session.llm_failures >= 2
                and current_state == AssistantState.LISTENING
                and (session.llm_task is None or session.llm_task.done())
            ):
                await _end_call_with_line(policy.template("TECH_TROUBLE"), "TECH_TROUBLE", "LLM_FAILURE",
                                          cache_key="TECH_TROUBLE")
                break

            # Speech-to-text never connected: the caller can't be heard, so end politely
            # rather than waiting out the inactivity timeout in silence.
            if (
                stt_connect_task.done()
                and not stt_connect_task.cancelled()
                and stt_connect_task.exception() is None
                and stt_connect_task.result() is False
                and not getattr(session.stt_stream, "_connected", False)
                and current_state == AssistantState.LISTENING
                and (session.llm_task is None or session.llm_task.done())
            ):
                print(f"[Session {session_id}] [STT] Deepgram unavailable; ending call.")
                await _end_call_with_line(policy.template("TECH_TROUBLE"), "TECH_TROUBLE", "STT_UNAVAILABLE",
                                          cache_key="TECH_TROUBLE")
                break

            # Caller asked NORA to hold: check in once, then end after the hold limit.
            if session.hold_active and current_state == AssistantState.LISTENING and (
                session.llm_task is None or session.llm_task.done()
            ):
                hold = policy.hold_status(session)
                if hold == "CHECKIN":
                    session.hold_checkin_sent = True
                    await _say_line(policy.template("HOLD_CHECKIN"), "HOLD_CHECKIN", cache_key="HOLD_CHECKIN")
                    continue
                if hold == "TIMEOUT":
                    session.hold_active = False
                    await _end_call_with_line(policy.template("HOLD_TIMEOUT"), "HOLD_TIMEOUT", "HOLD_TIMEOUT",
                                              cache_key="HOLD_TIMEOUT")
                    break

            # End call after sustained post-TTS silence while listening.
            # Configurable via CALL_INACTIVITY_TIMEOUT_SECS (default 45).
            # Suspended while the caller has asked NORA to hold.
            if (
                current_state == AssistantState.LISTENING
                and not session.hold_active
                and time.time() - session.last_active_time > inactivity_timeout_s
            ):
                print(
                    f"[Session {session_id}] {inactivity_timeout_s:.0f}s of silence detected after TTS ended. Ending call."
                )
                await play_cached_audio(
                    session, "inactivity_bye", ws_audio_queue,
                    fallback_text="Hey, looks like you might've stepped away. No worries, have a good one!",
                )
                session.final_disposition = session.final_disposition or "NO_RESPONSE"
                # play_cached_audio already waited for playback to finish.
                await asyncio.sleep(END_CALL_GRACE_SECS)
                await websocket.close(code=1000)
                break

            # 1. Enforce max call duration (3 minutes)
            if session.timed_out():
                print(f"[Session {session_id}] Session timed out after {session.timeout_secs}s.")
                await _force_stop_outbound_audio("TIMEOUT")

                # Cancel any in-flight LLM pipeline so it doesn't race with our wrap-up
                if session.llm_task and not session.llm_task.done():
                    session.llm_task.cancel()
                    try:
                        await session.llm_task
                    except asyncio.CancelledError:
                        pass
                    session.llm_task = None

                # Allow a clean wrap-up pipeline after force-stopping stale audio.
                session.clear_interrupts()

                # Build a brief recap of recent exchanges so the LLM can
                # craft a goodbye that flows naturally from the conversation.
                recent_lines = []
                for msg in session.chat_history[-6:]:
                    role_label = "You" if msg["role"] == "assistant" else "Them"
                    recent_lines.append(f"{role_label}: {msg['content']}")
                recap = "\n".join(recent_lines) if recent_lines else "(no exchanges yet)"

                wrap_up_prompt = (
                    f"[SYSTEM — WRAP UP NOW]\n"
                    f"Here's where the conversation is at:\n{recap}\n\n"
                    f"You need to get off the phone. Give a very quick, natural, "
                    f"human reason to wrap up that fits the flow of the conversation above. "
                    f"If they seemed interested, say you'll follow up. If they were hesitant, "
                    f"just thank them for their time. Do NOT ask any questions. "
                    f"Do NOT mention any time limits. Keep it under 20 words."
                )

                async def _run_timeout_goodbye(_prompt: str, _bye_cache: str):
                    _exit_q: asyncio.Queue = asyncio.Queue()
                    _llm = asyncio.create_task(get_ai_response_stream(
                        _prompt, "en", session, _exit_q,
                        max_tokens=80, max_words=30,
                        tier="COMMAND", primary_intent="WRAP_UP_TIMEOUT",
                    ))
                    await process_tts_queue(
                        session, _exit_q, ws_audio_queue,
                        primary_intent="WRAP_UP_TIMEOUT",
                        tier="COMMAND", refusal_count=session.refusal_count,
                    )
                    if not _llm.done():
                        _llm.cancel()
                        try:
                            await _llm
                        except asyncio.CancelledError:
                            pass

                chunks_before = session.audio_chunks_sent
                try:
                    await asyncio.wait_for(
                        _run_timeout_goodbye(wrap_up_prompt, "timeout_bye"),
                        timeout=10.0,
                    )
                    await _ensure_goodbye_played(chunks_before, "timeout_bye")
                except asyncio.TimeoutError:
                    print(f"[Session {session_id}] Goodbye pipeline timed out — using cached audio.")
                    session.clear_interrupts()
                    await play_cached_audio(
                        session, "timeout_bye", ws_audio_queue,
                        fallback_text="Thanks so much for your time! Have a great day.",
                    )
                session.final_disposition = session.final_disposition or "MAX_DURATION"
                await asyncio.sleep(END_CALL_GRACE_SECS)
                await websocket.close(code=1000)
                break

            # 2. Enforce max conversation turns (12 turns)
            if session.turns_exceeded():
                print(f"[Session {session_id}] Turn limit reached ({session.MAX_TURNS} turns). Ending call.")
                await _force_stop_outbound_audio("TURN_LIMIT")

                # Cancel any in-flight LLM pipeline
                if session.llm_task and not session.llm_task.done():
                    session.llm_task.cancel()
                    try:
                        await session.llm_task
                    except asyncio.CancelledError:
                        pass
                    session.llm_task = None

                # Allow a clean wrap-up pipeline after force-stopping stale audio.
                session.clear_interrupts()

                # Build recap for context-aware goodbye
                recent_lines = []
                for msg in session.chat_history[-6:]:
                    role_label = "You" if msg["role"] == "assistant" else "Them"
                    recent_lines.append(f"{role_label}: {msg['content']}")
                recap = "\n".join(recent_lines) if recent_lines else "(no exchanges yet)"

                wrap_up_prompt = (
                    f"[SYSTEM — WRAP UP NOW]\n"
                    f"Here's where the conversation is at:\n{recap}\n\n"
                    f"You need to get off the phone. Give a very quick, natural, "
                    f"human reason to wrap up that fits the flow of the conversation above. "
                    f"If they seemed interested, say you'll follow up. If they were hesitant, "
                    f"just thank them for their time. Do NOT ask any questions. "
                    f"Do NOT mention any turn limits. Keep it under 20 words."
                )

                async def _run_turnlimit_goodbye(_prompt: str):
                    _exit_q: asyncio.Queue = asyncio.Queue()
                    _llm = asyncio.create_task(get_ai_response_stream(
                        _prompt, "en", session, _exit_q,
                        max_tokens=80, max_words=30,
                        tier="COMMAND", primary_intent="WRAP_UP_TIMEOUT",
                    ))
                    await process_tts_queue(
                        session, _exit_q, ws_audio_queue,
                        primary_intent="WRAP_UP_TIMEOUT",
                        tier="COMMAND", refusal_count=session.refusal_count,
                    )
                    if not _llm.done():
                        _llm.cancel()
                        try:
                            await _llm
                        except asyncio.CancelledError:
                            pass

                chunks_before = session.audio_chunks_sent
                try:
                    await asyncio.wait_for(
                        _run_turnlimit_goodbye(wrap_up_prompt),
                        timeout=10.0,
                    )
                    await _ensure_goodbye_played(chunks_before, "turn_limit_bye")
                except asyncio.TimeoutError:
                    print(f"[Session {session_id}] Turn-limit goodbye timed out — using cached audio.")
                    session.clear_interrupts()
                    await play_cached_audio(
                        session, "turn_limit_bye", ws_audio_queue,
                        fallback_text="Thanks so much for your time! Have a great day.",
                    )
                session.final_disposition = session.final_disposition or "MAX_TURNS"
                await asyncio.sleep(END_CALL_GRACE_SECS)
                await websocket.close(code=1000)
                break
                
            # 2. Receive incoming audio chunk (16kHz PCM 16-bit mono)
            try:
                # Use a small timeout to allow checking session states/timeouts periodically
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
            except asyncio.TimeoutError:
                # If we're starved of audio chunks but Deepgram is still transcribing,
                # keep the session alive to prevent the 30s timeout from triggering.
                if session.stt_stream and (len(session.stt_stream._final_parts) > 0 or len(session.stt_stream._latest_interim) > 0):
                    session.last_active_time = time.time()
                continue

            # Ensure we only process even-length bytes to avoid 'buffer size must be a multiple of element size'
            if len(data) % 2 != 0:
                print(f"[Session {session_id}] Received odd-length byte chunk: {len(data)}")
                data = data[:-1]
                
            if not data:
                continue

            # Convert bytes to normalized float32 tensor
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # For VAD we need a tensor
            tensor = torch.from_numpy(audio_array)
            if len(tensor) == 0:
                continue

            # ── Send ALL audio to Deepgram immediately (before VAD) ──
            # This ensures zero audio loss regardless of VAD buffer state.
            # Uses cached current_state from top of iteration — no extra lock.
            if current_state == AssistantState.LISTENING and session.stt_stream:
                await session.stt_stream.send_audio(data)
            
            # VAD requires exactly 512 samples per inference chunk
            vad_buffer.append(audio_array)
            vad_samples_accumulated += len(audio_array)
            
            # Only run VAD when we have enough samples for a proper read
            while vad_samples_accumulated >= 512:
                # Optimize: skip np.concatenate if buffer is a single array
                if len(vad_buffer) == 1:
                    combined_audio = vad_buffer[0]
                else:
                    combined_audio = np.concatenate(vad_buffer)
                vad_tensor = torch.from_numpy(combined_audio[:512])
                
                # Keep leftover for the next cycle
                leftover = combined_audio[512:]
                vad_buffer = [leftover] if len(leftover) > 0 else []
                vad_samples_accumulated = len(leftover)
                
                # Reuse cached state — no extra async lock acquisition
                state = current_state
                
                # 3. Interruption detection if assistant is speaking
                if state == AssistantState.SPEAKING:
                    interrupted = await vad_service.detect_interruption(session, vad_tensor)
                    
                    if interrupted:
                        print(f"[Session {session_id}] [INTERRUPTION] Stopped TTS. Now listening to user.")
                        
                        # Drain the sender queue so we don't send stale audio to the browser
                        # BUG 9 FIX: use atomic get_nowait loop (not empty() which has a TOCTOU race)
                        while True:
                            try:
                                ws_audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                                
                        # Tell frontend to stop playing its TTS buffer immediately
                        # Send generation ID so frontend can gate future chunks
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "CLEAR",
                                "generation": session.tts_generation
                            }))
                        except Exception as e:
                            pass
                            
                        # Ensure the active LLM background pipeline actually halts right now
                        # (instead of just setting an event stringing it along)
                        if session.llm_task and not session.llm_task.done():
                            session.llm_task.cancel()
                            try:
                                await session.llm_task
                            except asyncio.CancelledError:
                                pass
                            session.llm_task = None
                        
                        # Keep Deepgram stream open; only clear transcript state and
                        # forward interruption-buffer audio so first user words are preserved.
                        if session.stt_stream:
                            await session.stt_stream.reset_transcript()
                            for buf_chunk in session.interruption_buffer:
                                await session.stt_stream.send_audio_float32(buf_chunk)
                            
                        # BUG 8 FIX: seed with a small constant instead of buffer length.
                        # interruption_buffer holds 512-sample numpy chunks, not WebSocket
                        # receive frames, so using its length broke the > 10 speech guard.
                        audio_chunk_count = 3
                        session.interruption_buffer = []
                        
                        silence_frames = 0
                        session.has_speech_in_buffer = True

                    continue
                
                # 4. Listening and accumulating speech
                if state == AssistantState.LISTENING:
                    # Basic VAD check to detect silence end-of-speech
                    # Run in thread pool to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    # _ensure_model() lazy-loads once; if the model is unavailable,
                    # fall back to Deepgram endpointing instead of crashing the call.
                    vad_model = vad_service.model or vad_service._ensure_model()
                    if vad_model is None:
                        prob = 0.0
                    else:
                        with torch.no_grad():
                            prob = await loop.run_in_executor(
                                _vad_executor, lambda t=vad_tensor, m=vad_model: float(m(t, 16000).item())
                            )
                        
                    # Also consider Deepgram's own transcript state. If Deepgram heard speech,
                    # we must treat the user as active even if local VAD probability is low.
                    dg_has_speech = False
                    if session.stt_stream:
                        dg_has_speech = len(session.stt_stream._final_parts) > 0 or len(session.stt_stream._latest_interim) > 0

                    if prob < 0.25 and not dg_has_speech:  # Looser threshold for identifying absolute silence
                        silence_frames += 1
                    else:
                        silence_frames = 0
                        session.last_active_time = time.time()
                        session.has_speech_in_buffer = True
                    
                    audio_chunk_count += 1
            if vad_samples_accumulated < 512:
                # If we don't have 512 samples yet and state is SPEAKING, skip accumulation
                if current_state == AssistantState.SPEAKING:
                    continue  # We don't need to accumulate speech buffer while speaking, just wait for VAD

            # Deepgram endpointing fallback — check this BEFORE the if/elif chain.
            # When Deepgram has finals + 0.8s of quiet, we go directly to turn
            # processing instead of waiting for local VAD silence (which can't
            # increment when dg_has_speech is True).
            dg_endpointing_ready = (
                session.stt_stream
                and (
                    # Approach A: Finals + short silence (tightened from 0.3)
                    (session.stt_stream.has_any_final()
                     and session.stt_stream.seconds_since_last_result() > 0.2)
                    # Approach B: Deepgram signalled utterance end AND we have words.
                    # The flag is sticky until the transcript is reset, so without
                    # this guard every later audio chunk would end an empty "turn".
                    or (session.stt_stream.utterance_end_received()
                        and session.stt_stream.has_any_transcript())
                    # Approach C: final segment flagged speech_final — Deepgram's
                    # endpointer already waited for silence, so skip the extra 0.2 s.
                    or (session.stt_stream.has_any_final()
                        and session.stt_stream.speech_final_received())
                )
                and audio_chunk_count > 10
            )

            # 5. Evaluate End of Turn (Silence detected, proceed to transcription)
            if silence_frames > WIPE_BUFFER_SILENCE_FRAMES and audio_chunk_count <= 10:
                # User made a tiny noise (cough/breath) but stayed silent for 1.5+ seconds.
                # It's a false positive. Wipe the buffer before it eats RAM endlessly.
                if audio_chunk_count > 0:
                    print(f"[Session {session_id}] [VAD] Wiping false-positive noise buffer ({audio_chunk_count} chunks).")
                audio_chunk_count = 0
                session.has_speech_in_buffer = False
                # We do NOT reset silence_frames here so the 15s absolute silence timeout can still trigger if they went AFK.

            elif (silence_frames > SILENCE_THRESHOLD_FRAMES and audio_chunk_count > 10) or dg_endpointing_ready:
                
                # If triggered by Deepgram endpointing, ensure speech flag is set
                if dg_endpointing_ready:
                    session.has_speech_in_buffer = True

                # Check if Deepgram caught anything despite local VAD thinking the buffer is empty
                dg_has_speech = False
                if session.stt_stream and not session.has_speech_in_buffer:
                    dg_has_speech = session.stt_stream.has_any_transcript()

                if session.has_speech_in_buffer or dg_has_speech:
                    await session.set_state(AssistantState.THINKING)
                    session.has_speech_in_buffer = False
                    # Latency marks for this turn are measured from end-of-turn detection.
                    session.turn_metrics = {}
                    session.mark_turn("speech_end")

                    # ── Instant transcript from Deepgram streaming ──
                    # By now Deepgram has been receiving audio in real-time,
                    # so the transcript is already available — no network wait.
                    transcript = ""
                    if session.stt_stream:
                        transcript = await session.stt_stream.get_final_transcript()
                    session.mark_turn("transcript")
                    
                    # Reset buffers
                    audio_chunk_count = 0
                    silence_frames = 0
                    
                    if transcript:
                        safe_transcript = transcript.encode('ascii', 'replace').decode('ascii')
                        print(f"\nUser spoken transcribed text: {safe_transcript}\n")
                        print(f"[Session {session_id}] Transcript: {safe_transcript}")

                        # Deepgram can occasionally surface an interim-fallback text and then
                        # a near-immediate final text with identical content. Ignore exact
                        # duplicates in a short window so we do not double-process one utterance.
                        normalized_transcript = re.sub(r"\s+", " ", transcript.strip().lower())
                        now_ts = time.time()
                        if (
                            normalized_transcript
                            and normalized_transcript == session.last_transcript_normalized
                            and (now_ts - session.last_transcript_time) < 2.5
                        ):
                            print(f"[Session {session_id}] [STT] Ignored duplicate transcript within 2.5s window.")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            continue

                        # Ignore ultra-short interruption fragments that are not real answers.
                        short_tokens = normalized_transcript.split()
                        if len(short_tokens) == 1 and short_tokens[0] in {
                            "i", "you", "uh", "um", "hmm", "mm", "mhm", "uhh", "erm", "but"
                        }:
                            print(f"[Session {session_id}] [STT] Ignored short fragment: {safe_transcript}")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            continue

                        # Short control/compliance phrases ("one sec", "bye", "what?",
                        # "stop calling") are real turns, never fragments.
                        # Short yes/no answers ("yes speaking", "sure thing", "nope") too.
                        short_is_meaningful = (
                            len(short_tokens) <= 2
                            and (
                                heuristic_intent_for(transcript) != OutreachIntent.UNCLEAR
                                or policy.is_affirmative(transcript)
                                or policy.is_negative(transcript)
                            )
                        )
                        if (
                            len(short_tokens) <= 2
                            and not short_is_meaningful
                            and _last_assistant_asked_question(session)
                            and not _looks_like_short_benefit_answer(normalized_transcript)
                            and not _looks_like_short_identifier_answer(normalized_transcript)
                            and not _looks_like_short_contact_answer(transcript)
                        ):
                            print(f"[Session {session_id}] [STT] Ignored short answer fragment: {safe_transcript}")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            continue

                        if (
                            _looks_like_incomplete_benefit_answer(normalized_transcript)
                            and heuristic_intent_for(transcript) == OutreachIntent.UNCLEAR
                        ):
                            print(f"[Session {session_id}] [STT] Ignored incomplete phrase: {safe_transcript}")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            continue

                        if _looks_like_partial_date_fragment(normalized_transcript):
                            print(f"[Session {session_id}] [STT] Ignored partial date fragment: {safe_transcript}")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            continue

                        # Voicemail greeting or IVR menu at the start of the call:
                        # leave a limited-content message (or hang up on an IVR).
                        telephony_kind = (
                            policy.classify_telephony_prompt(normalized_transcript)
                            if session.turn_count == 0 else None
                        )
                        if telephony_kind:
                            print(f"[Session {session_id}] [STT] {telephony_kind} detected: {safe_transcript}")
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            session.clear_interrupts()
                            if telephony_kind == "VOICEMAIL":
                                # Let the beep pass before speaking.
                                await asyncio.sleep(1.0)
                                await _end_call_with_line(
                                    policy.template("VOICEMAIL", org=policy._org(session)),
                                    "VOICEMAIL_LEFT", "VOICEMAIL",
                                )
                            else:
                                session.final_disposition = "IVR_REACHED"
                                await _force_stop_outbound_audio("IVR")
                                await websocket.close(code=1000)
                            break

                        # Ignore early carrier/voicemail bridge announcements so
                        # system audio is not treated as user intent.
                        if session.turn_count == 0 and _looks_like_telephony_prompt(normalized_transcript):
                            print(
                                f"[Session {session_id}] [STT] Ignored telephony prompt transcript: {safe_transcript}"
                            )
                            session.clear_interrupts()
                            if session.stt_stream:
                                await session.stt_stream.reset_transcript()
                            await session.set_state(AssistantState.LISTENING)
                            if (
                                greeting_text
                                and greeting_cache_name
                                and not greeting_replayed_once
                            ):
                                greeting_replayed_once = True
                                print(f"[Session {session_id}] [GREETING] Replaying greeting after telephony prompt.")
                                asyncio.create_task(play_cached_audio(
                                    session,
                                    greeting_cache_name,
                                    ws_audio_queue,
                                    fallback_text=greeting_text,
                                    expect_text=greeting_text,
                                ))
                            continue

                        if session.stt_stream:
                            await session.stt_stream.reset_transcript()
                        session.last_transcript_normalized = normalized_transcript
                        session.last_transcript_time = now_ts
                        
                        lang = detect_language(transcript)
                        session.last_user_input = transcript
                        session.clear_interrupts()
                        
                        # ── Increment turn counter ──
                        session.turn_count += 1
                        print(f"[Session {session_id}] [TURN] {session.turn_count}/{session.MAX_TURNS}")

                        vob_mode = bool(
                            session.lead_context
                            and (
                                str(session.lead_context.get("call_type", "")).upper() == "VOB"
                                or session.lead_context.get("vob_mode")
                            )
                        )

                        # ── Fast path: pending email confirmation ──
                        # If we are only waiting for a yes/no to confirm a known email,
                        # skip full intent classification to avoid avoidable latency.
                        if not vob_mode and session.email_pending_confirmation and session.email_address:
                            quick_affirmative = bool(re.search(
                                r"\b(yes|yeah|yep|yup|sure|ok|okay|cool|alright|perfect|absolutely|correct|exactly|"
                                r"go\s*ahead|please\s*send|send\s*it|do\s*that|we'?ll\s*do\s*that|"
                                r"go\s*for\s*it|sounds\s*good|that\s*works|you\s*can|"
                                r"that'?s\s*(right|correct|true|it|my\s*email|fine|good|okay|ok)|"
                                r"that\s+is\s*(right|correct|true|it|my\s*email)|"
                                r"it\s+is)\b",
                                transcript, re.IGNORECASE,
                            ))
                            quick_negative = bool(re.search(
                                r"\b(no(?!\s*(?:problem|worries|doubt))|wrong|incorrect|nope|nah|not\s*right|change\s*it|mistake|"
                                r"not\s+looking|not\s+interested|not\s+now|too\s+busy)\b",
                                transcript, re.IGNORECASE,
                            ))

                            if quick_affirmative and not quick_negative:
                                session.email_captured = True
                                session.email_pending_confirmation = False
                                session.outcome = session.outcome or "WRITTEN_DETAILS"
                                print(f"[Session {session_id}] [EMAIL] Confirmed quickly: {session.email_address}")
                                await _end_call_with_line(
                                    "Okay, I'll send over the details by email. "
                                    "Thank you for your time. Have a great day, goodbye.",
                                    "WRITTEN_DETAILS_REQUESTED", "EMAIL_CONFIRM_FAST", user_text=transcript,
                                )
                                break
                            elif quick_negative:
                                print(f"[Session {session_id}] [EMAIL] Quick negative confirmation. Clearing pending email: {session.email_address}")
                                session.email_address = None
                                session.email_pending_confirmation = False
                                if session.lead_context and 'email' in session.lead_context:
                                    session.lead_context['email'] = None

                        # ── Fallback: email confirm context drift ──
                        # If pending flag is unexpectedly false but we recently asked to
                        # confirm a known email and user says "yes/it is", end gracefully.
                        if not vob_mode and session.email_address and not session.email_captured and not session.email_pending_confirmation:
                            last_assistant = ""
                            for msg in reversed(session.chat_history):
                                if msg.get("role") == "assistant":
                                    last_assistant = (msg.get("content") or "").lower()
                                    break

                            asked_email_confirmation = (
                                "best place for that" in last_assistant
                                or "best email" in last_assistant
                                or "still the best place" in last_assistant
                            )
                            fallback_affirmative = bool(re.search(
                                r"\b(yes|yeah|yep|yup|sure|ok|okay|correct|exactly|that'?s\s*(right|correct|it)|it\s+is)\b",
                                transcript, re.IGNORECASE,
                            ))
                            fallback_negative = bool(re.search(
                                r"\b(no(?!\s*(?:problem|worries|doubt))|wrong|incorrect|nope|nah|not\s*right|change\s*it|mistake)\b",
                                transcript, re.IGNORECASE,
                            ))

                            if asked_email_confirmation and fallback_affirmative and not fallback_negative:
                                session.email_captured = True
                                session.outcome = session.outcome or "WRITTEN_DETAILS"
                                print(f"[Session {session_id}] [EMAIL] Confirmed via fallback context: {session.email_address}")
                                await _end_call_with_line(
                                    "Okay, I'll send over the details by email. "
                                    "Thank you for your time. Have a great day, goodbye.",
                                    "WRITTEN_DETAILS_REQUESTED", "EMAIL_CONFIRM_FALLBACK", user_text=transcript,
                                )
                                break
                        
                        # ── Parallel Intent classification & Speculative LLM ──
                        intent_task = asyncio.create_task(classify_intent(transcript, turn_count=session.turn_count))
                        
                        speculative_intent, speculative_tier = _predict_speculative_intent_and_tier(transcript)
                        # Only speculate when we have a concrete intent guess.
                        # Keep speculative path filler-free to avoid random cached
                        # fillers playing before intent is finalized.
                        speculative_task = None
                        speculative_filler_dispatched = False
                        # Never speculate while the policy engine is mid-flow (waiting
                        # for a callback time, or on hold): its reply is deterministic.
                        speculative_enabled = (
                            speculative_intent != "UNCLEAR"
                            and speculative_tier in {"COMMAND", "OBJECTION", "DEFINITION"}
                            and not session.pending_callback_reason
                            and not session.hold_active
                        )
                        if speculative_enabled:
                            # Use tone-based filler instead of intent-based.
                            # The speculative intent is a rough regex guess that's
                            # often wrong, leading to mismatched fillers like
                            # "Gotcha," for questions.  Tone analysis on the raw
                            # transcript is more reliable for filler selection.
                            speculative_filler = get_speculative_filler(transcript)
                            speculative_filler_dispatched = bool(speculative_filler)
                            speculative_task = asyncio.create_task(run_llm_pipeline(
                                transcript, lang, session, ws_audio_queue,
                                max_tokens=180,
                                max_words=60,
                                tier=speculative_tier,
                                primary_intent=speculative_intent,
                                filler_text=speculative_filler,
                                filler_used=bool(speculative_filler),
                                current_mode=session.current_mode,
                                pitch_delivered=session.pitch_delivered,
                            ))
                            session.llm_task = speculative_task
                        else:
                            session.llm_task = None

                        intent = await intent_task
                        session.mark_turn("intent")
                        if (
                            vob_mode
                            and _last_assistant_was_benefit_question(session)
                            and _looks_like_short_benefit_answer(transcript)
                        ):
                            intent.primary_intent = OutreachIntent.UNCLEAR
                            intent.tier = ContentTier.COMMAND
                            intent.counts_as_refusal = False
                            intent.is_exit = False
                        if (
                            vob_mode
                            and _last_assistant_was_benefit_question(session)
                            and _looks_like_benefit_response(transcript)
                            and not _looks_like_clear_refusal(transcript)
                        ):
                            intent.primary_intent = OutreachIntent.UNCLEAR
                            intent.tier = ContentTier.COMMAND
                            intent.counts_as_refusal = False
                            intent.is_exit = False
                        session.record_intent(intent.primary_intent.value)
                        session.record_mood(intent.primary_intent.value)

                        # Track how many times the name could not be understood.
                        # FAKE_NAME counts too — if they gave a joke name that's also a failed attempt.
                        if intent.primary_intent.value in {"CLARIFY_NAME", "FAKE_NAME"}:
                            session.name_clarify_attempts += 1
                            print(f"[Session {session_id}] [NAME] Clarify attempt #{session.name_clarify_attempts}")
                        print(
                            f"[Session {session_id}] [INTENT] "
                            f"exit={intent.is_exit}, intent={intent.primary_intent.value}, "
                            f"refusal={intent.counts_as_refusal}, tier={intent.tier.value}, "
                            f"latency={intent.latency_ms:.0f}ms"
                        )
                        
                        # Decide if speculative output can be kept.
                        if speculative_task is None:
                            needs_restart = True
                        else:
                            intent_mismatch = intent.primary_intent.value != speculative_intent
                            tier_mismatch = intent.tier.value != speculative_tier
                            needs_restart = (
                                intent.is_exit or
                                intent.counts_as_refusal or
                                tier_mismatch or
                                intent_mismatch or
                                session.email_pending_confirmation
                            )
                        
                        if needs_restart:
                            if speculative_task is not None and session.llm_task == speculative_task and not speculative_task.done():
                                session.interrupt_llm.set()
                                session.interrupt_tts.set()
                                speculative_task.cancel()
                                try:
                                    await speculative_task
                                except asyncio.CancelledError:
                                    pass
                                session.llm_task = None
                                session.clear_interrupts()

                        if intent.counts_as_refusal:
                            refusal_count = session.record_refusal(intent.primary_intent.value)
                            print(f"[Session {session_id}] [REFUSAL] Count={refusal_count} (Hard: {session.hard_refusal_count}, Soft: {session.soft_refusal_count})")
                            
                            # Tell the frontend about the refusal so the UI updates
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "REFUSAL_UPDATE",
                                    "refusalCount": refusal_count,
                                    "maxRefusals": 3,
                                    "intent": intent.primary_intent.value,
                                    "transcript": safe_transcript,
                                    "timestamp": int(time.time() * 1000)
                                }))
                            except Exception as e:
                                print(f"[Session {session_id}] Failed to send refusal update to client: {e}")
                        else:
                            refusal_count = session.refusal_count

                        # ── Policy engine: deterministic call-flow decisions ──
                        # Compliance stops, safety, escalation-to-callback, soft
                        # objections, hold/repeat/slow-down and AI disclosure are
                        # answered with fixed lines; everything else goes to the LLM.
                        action = policy.decide(session, transcript, intent.primary_intent.value)
                        if action.kind != policy.LLM:
                            if speculative_task is not None and not speculative_task.done():
                                session.interrupt_llm.set()
                                session.interrupt_tts.set()
                                speculative_task.cancel()
                                try:
                                    await speculative_task
                                except asyncio.CancelledError:
                                    pass
                                session.llm_task = None
                                session.clear_interrupts()
                            await _persist_policy_records(action)
                            if action.kind == policy.END:
                                await _end_call_with_line(
                                    action.text, action.disposition, action.reason, user_text=transcript,
                                    cache_key=action.cache_key,
                                )
                                break
                            await _cancel_llm_task()
                            session.clear_interrupts()
                            await _say_line(action.text, action.reason, user_text=transcript,
                                            cache_key=action.cache_key)
                            continue

                        # ── Email collection flow ──
                        if not vob_mode:
                            # Pending confirmations were resolved by the fast path above.
                            # 1. Pre-fill the email on file when the caller asks for written details.
                            if (
                                intent.primary_intent.value == "EMAIL_REQUEST"
                                and not session.email_captured
                                and not session.email_pending_confirmation
                                and session.identity_verified
                            ):
                                lead_email = session.get_lead_email()
                                if lead_email:
                                    session.email_address = lead_email
                                    session.email_pending_confirmation = True
                                    print(f"[Session {session_id}] [EMAIL] Pre-filled from lead context: {lead_email}")
                                    # The caller must hear the email read back and confirm on a separate turn.

                            # 2. Opportunistically capture any email address spoken or spelled out.
                            #    Handles both direct format (user@domain.com) and spoken format
                            #    ("habiba dot imran at gmail dot com").
                            if not session.email_captured:
                                email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", transcript, re.IGNORECASE)
                                if not email_match:
                                    # STT transcribes emails as spoken words, e.g.
                                    # "habiba dot imran at gmail dot com"
                                    # "habiba dot imran gmail dot com" (no "at")
                                    # "sarah at mitchell clinic dot com"
                                    spoken_match = re.search(
                                        r"([a-z0-9]+(?:\s+dot\s+[a-z0-9]+)*)"         # user part: "habiba dot imran"
                                        r"\s+(?:at|at the rate(?:\s+of)?)\s+"          # @ separator
                                        r"([a-z0-9]+(?:\s+[a-z0-9]+)*)"               # domain: "gmail" or "mitchell clinic"
                                        r"\s+dot\s+"                                    # .
                                        r"(com|org|net|co|io|edu|gov|health|clinic|us|info|biz|me)",   # TLD
                                        transcript, re.IGNORECASE,
                                    )
                                    # Fallback: user skipped "at" but used a well-known domain
                                    # e.g. "habiba dot imran gmail dot com"
                                    if not spoken_match:
                                        spoken_match = re.search(
                                            r"([a-z0-9]+(?:\s+dot\s+[a-z0-9]+)*)"     # user part
                                            r"\s+"                                      # space (no "at")
                                            r"(gmail|yahoo|hotmail|outlook|aol|icloud|protonmail|zoho)"  # known domain
                                            r"\s+dot\s+"
                                            r"(com|org|net|co|io|edu|gov|health|clinic|us|info|biz|me)",
                                            transcript, re.IGNORECASE,
                                        )
                                    if spoken_match:
                                        user_part = spoken_match.group(1).lower().replace(" dot ", ".").replace(" ", "")
                                        domain_part = spoken_match.group(2).lower().replace(" ", "")
                                        tld_part = spoken_match.group(3).lower()
                                        reconstructed = f"{user_part}@{domain_part}.{tld_part}"
                                        session.email_address = reconstructed
                                        session.email_pending_confirmation = True
                                        print(f"[Session {session_id}] [EMAIL] Captured (spoken): {reconstructed}")
                                else:
                                    session.email_address = email_match.group(0)
                                    session.email_pending_confirmation = True
                                    print(f"[Session {session_id}] [EMAIL] Captured (direct): {session.email_address}")

                            # A newly captured email must be read back, so a speculative
                            # reply that didn't know about it can't be kept.
                            if session.email_pending_confirmation and speculative_task is not None and not needs_restart:
                                needs_restart = True
                                if session.llm_task == speculative_task and not speculative_task.done():
                                    session.interrupt_llm.set()
                                    session.interrupt_tts.set()
                                    speculative_task.cancel()
                                    try:
                                        await speculative_task
                                    except asyncio.CancelledError:
                                        pass
                                    session.llm_task = None
                                    session.clear_interrupts()

                        # ── Immediate exit intents (fallback) ──
                        # The policy engine ends these calls with a fixed line; this only
                        # runs if an exit intent ever reaches here without a policy action.
                        if intent.is_exit:
                            exit_intent = intent.primary_intent.value
                            print(f"[Session {session_id}] Immediate exit intent (fallback): {exit_intent}")
                            session.clear_interrupts()
                            exit_tts_queue = asyncio.Queue()
                            llm_exit_task = asyncio.create_task(get_ai_response_stream(
                                transcript, lang, session, exit_tts_queue,
                                max_tokens=intent.max_tokens,
                                max_words=intent.max_words,
                                tier=intent.tier.value,
                                primary_intent=exit_intent,
                            ))
                            await process_tts_queue(
                                session,
                                exit_tts_queue,
                                ws_audio_queue,
                                primary_intent=exit_intent,
                                tier=intent.tier.value,
                                refusal_count=session.refusal_count,
                            )

                            try:
                                await asyncio.wait_for(llm_exit_task, timeout=2.0)
                            except asyncio.TimeoutError:
                                pass

                            session.final_disposition = session.final_disposition or exit_intent
                            await asyncio.sleep(END_CALL_GRACE_SECS)
                            await websocket.close(code=1000)
                            break

                        if session.should_end_after_refusal():
                            session.clear_interrupts()
                            if session.hard_refusal_count > 1:
                                print(f"[Session {session_id}] Repeated hard refusal — ending call.")
                                await play_cached_audio(
                                    session, "hard_decline_bye", ws_audio_queue,
                                    fallback_text="Apologies for the inconvenience. Have a nice day.",
                                )
                            else:
                                print(f"[Session {session_id}] Third refusal reached — ending call respectfully.")
                                await play_cached_audio(
                                    session, "decline_bye", ws_audio_queue,
                                    fallback_text="No worries at all. Thanks for taking the call. Have a good day!",
                                )
                            session.final_disposition = session.final_disposition or "DECLINED"
                            # play_cached_audio already waited for playback to finish.
                            await asyncio.sleep(END_CALL_GRACE_SECS)
                            await websocket.close(code=1000)
                            break

                        # Not an exit — start the LLM pipeline as a background task.
                        # MUST NOT await here — the main loop needs to keep receiving
                        # audio frames so VAD/interruption detection stays active
                        # during TTS playback. run_llm_pipeline manages its own
                        # state transitions (SPEAKING → LISTENING) internally.
                        
                        if needs_restart:
                            # ── Pick a thinking filler ──
                            # Play on every turn so there is no dead air while the LLM
                            # generates. COMMAND tier → "Umm,", "Hmm,", etc.
                            # EXPLANATION → "Good question.", "Happy to explain." etc.
                            # Intent-specific contexts (objection, gatekeeper, …) are
                            # resolved inside get_thinking_filler automatically.
                            #
                            # IMPORTANT: If the speculative path already dispatched a
                            # filler (which plays from cache in ~0ms), do NOT play a
                            # second one here. The user would hear stuttered back-to-back
                            # fillers like "Gotcha— Hmm, ..." otherwise.
                            if speculative_filler_dispatched:
                                filler = ""
                            else:
                                if vob_mode and _looks_like_benefit_response(normalized_transcript):
                                    filler = get_speculative_filler(transcript)
                                else:
                                    filler = get_thinking_filler(
                                        primary_intent=intent.primary_intent.value,
                                        tier=intent.tier.value,
                                        refusal_count=session.refusal_count,
                                    )

                            session.llm_task = asyncio.create_task(run_llm_pipeline(
                                transcript, lang, session, ws_audio_queue,
                                max_tokens=intent.max_tokens,
                                max_words=intent.max_words,
                                tier=intent.tier.value,
                                primary_intent=intent.primary_intent.value,
                                filler_text=filler,
                                filler_used=bool(filler) or speculative_filler_dispatched,
                                current_mode=session.current_mode,
                                pitch_delivered=session.pitch_delivered,
                            ))

                        # Only count genuine engagement intents as meeting attempts.
                        # Refusal intents were previously included here, which inflated
                        # meeting_ask_count and caused the AI to stop asking for meetings
                        # too early in the conversation.
                        meeting_intents = {
                            "INTERESTED",
                            "ASK_MORE_INFO",
                            "ASK_AUDIT_DEFINITION",
                            "ASK_AUDIT_PROCESS",
                            "ASK_COMPLIANCE_BAA",
                            "CALL_BACK_LATER",
                            "EMAIL_REQUEST",
                        }
                        if intent.primary_intent.value in meeting_intents:
                            session.note_meeting_attempt()
                    else:
                        # Empty transcript (noise): clear Deepgram's end-of-utterance
                        # flag too, otherwise it re-triggers on the next audio chunk
                        # and can capture a half-finished interim as a turn.
                        if session.stt_stream:
                            await session.stt_stream.reset_transcript()
                        await session.set_state(AssistantState.LISTENING)
                else:
                    # Pure silence was captured due to buffer length reaching limit without speech
                    # Just clear the buffer to prevent memory bloat
                    audio_chunk_count = 0
                    silence_frames = 0

    except WebSocketDisconnect as e:
        print(f"[Session {session_id}] Disconnected by client. close_code={getattr(e, 'code', 'unknown')}")
    except Exception as e:
        print(f"[Session {session_id}] Unexpected error: {e}")
        traceback.print_exc()
        # The caller is probably still on the line: say goodbye instead of dropping them.
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await asyncio.wait_for(
                    _end_call_with_line(policy.template("TECH_TROUBLE"), "TECH_TROUBLE", "SERVER_ERROR",
                                        cache_key="TECH_TROUBLE"),
                    timeout=12.0,
                )
            except Exception:
                traceback.print_exc()
    finally:
        # Instantly halt any running LLM stream and TTS queue processing
        if session:
            session.interrupt_llm.set()
            session.interrupt_tts.set()

        sender_task.cancel()
        # Clean up the persistent TTS WebSocket connection
        await close_tts_connection(session)
        # Clean up the Deepgram streaming STT connection
        if session and session.stt_stream:
            await session.stt_stream.close()
            session.stt_stream = None
        print(f"[Session {session_id}] Disconnected and cleaned up.")

        # Persist a lightweight session summary row for analytics / later DB sync.
        try:
            if session:
                await log_session_summary(session)
        except Exception:
            traceback.print_exc()

        # BUG 1 FIX: Schedule session removal after a short grace window.
        # This allows brief reconnects to reuse the session while still
        # preventing session_store from growing unbounded across many calls.
        async def _expire_session():
            try:
                await asyncio.sleep(30)
                session_store.pop(session_id, None)
                print(f"[Session {session_id}] Expired from session store.")
            except asyncio.CancelledError:
                pass
                
        task = asyncio.create_task(_expire_session())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
