import asyncio
import os
import time
import pathlib
import re
from dotenv import load_dotenv

from app.session import AssistantState, Session
from app.utils.language import enhance_tts_text, split_response_for_tts, _resolve_emotion_style

load_dotenv()
RIME_API_KEY = os.getenv("RIME_API_KEY")


def _get_rime_time_scale_factor() -> float:
    """Read and clamp time scale factor (aka speed alpha) safely."""
    raw = os.getenv("RIME_TIME_SCALE_FACTOR", os.getenv("RIME_SPEED_ALPHA", "1.0")).strip()
    try:
        factor = float(raw)
    except ValueError:
        return 1.0
    return max(0.6, min(1.6, factor))


def _get_rime_repetition_penalty() -> float:
    """Read and clamp repetition penalty so misconfigured env values are harmless."""
    raw = os.getenv("RIME_REPETITION_PENALTY", "1.1").strip()
    try:
        penalty = float(raw)
    except ValueError:
        return 1.1
    return max(0.8, min(2.0, penalty))


def _get_rime_sampling_rate() -> int:
    """Read and clamp sampling rate to common 16kHz defaults."""
    raw = os.getenv("RIME_SAMPLING_RATE", "16000").strip()
    try:
        rate = int(raw)
    except ValueError:
        return 16000
    return 16000 if rate <= 0 else rate


def _get_rime_accept_header() -> str:
    """Map short audio format names to the Accept header Rime expects."""
    raw = os.getenv("RIME_AUDIO_FORMAT", "wav").strip().lower()
    if raw.startswith("audio/"):
        return raw
    mapping = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mp3",
        "opus": "audio/webm;codecs=opus",
        "ogg": "audio/ogg;codecs=opus",
        "mulaw": "audio/x-mulaw",
    }
    return mapping.get(raw, "audio/wav")

# Cached HTTP client (lazy loaded)
_http_client = None

# Rime voice and model config
RIME_VOICE = os.getenv("RIME_VOICE", "orion").strip()
RIME_MODEL = os.getenv("RIME_MODEL", "arcanav2").strip()
RIME_GENRE = os.getenv("RIME_GENRE", "conversational").strip()
RIME_AUDIO_FORMAT = os.getenv("RIME_AUDIO_FORMAT", "wav").strip().lower()
RIME_LANG = os.getenv("RIME_LANG", "").strip()
RIME_SAMPLING_RATE = _get_rime_sampling_rate()
RIME_TIME_SCALE_FACTOR = _get_rime_time_scale_factor()
RIME_REPETITION_PENALTY = _get_rime_repetition_penalty()
RIME_ACCEPT_HEADER = _get_rime_accept_header()
# Rime TTS API endpoint (streaming is not guaranteed; we simulate if needed)
_TTS_SYNC_URL = "https://users.rime.ai/v1/rime-tts"
_TTS_STREAM_URL = "https://users.rime.ai/v1/rime-tts"

# Pre-recorded SFX clips (16k PCM mono) for natural emotional cues.
_AUDIO_CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / "audio_cache"
_PCM_CHUNK_SIZE = 1600  # 100ms of 16kHz 16-bit mono — smaller chunks = faster first-byte delivery

# Laughter fallback pools — multiple phrase options so the agent doesn't repeat
# the same laugh cadence. Each phrase is a *complete conversational phrase* that
# embeds the amusement into natural speech pacing. TTS engines render these with
# proper prosody instead of trying to "perform" a standalone laugh clip.
_LAUGH_CHUCKLE_FALLBACKS = [
    "Ha, that's a fair point —",
    "Ha, okay I like that —",
    "Ha, okay okay —",
    "Ha, honestly —",
    "Ha, I'll give you that —",
]
_LAUGH_SOFT_FALLBACKS = [
    "Ha, honestly though —",
    "Ha, yeah no, that's funny —",
    "Ha, okay that's fair —",
    "Ha, sure sure —",
    "Ha, I hear you —",
    "Ha, yeah that tracks —",
]

import random as _sfx_random

def _pick_laugh_fallback(pool: list[str]) -> str:
    """Return a random fallback phrase from the pool."""
    return _sfx_random.choice(pool)

# Detects bare laughter openers that the LLM writes as text (e.g. "Ha, I think…").
# These sound robotic from TTS because the engine gets no prosodic context for a
# standalone laugh syllable. We intercept and route through the SFX pool instead,
# which renders a natural-sounding varied phrase ("Ha, that's fair —" etc.) via
# live TTS, then the rest of the sentence follows as a separate clean segment.
_HA_LAUGH_RE = re.compile(
    r"^(Ha[!,.]?|Haha[!,.]?|Hah[!,.]?)\s+",
    re.IGNORECASE,
)

_SFX_TOKEN_CONFIG = {
    # Laughter — use "POOL" sentinel; actual fallback is chosen at runtime
    # from _LAUGH_*_FALLBACKS so consecutive laughs sound different.
    "SFX_CHUCKLE":    ("laugh_chuckle.pcm", "POOL:CHUCKLE"),
    "SFX_LAUGH_SOFT": ("laugh_soft.pcm",    "POOL:LAUGH_SOFT"),
    "SFX_SIGH_SOFT":  ("sigh_soft.pcm",     "Hmm, yeah..."),
    # Non-verbal breath/filler — purely atmospheric; empty fallback = silent skip
    "SFX_BREATH_IN":  ("breath_in.pcm",     ""),
    "SFX_HMPH":       ("hmph.pcm",          "Hmm."),
    "SFX_WARMUP":     ("warmup.pcm",        "Yeah, so..."),
    # Expanded emotional cues
    "SFX_HMM":        ("hmm.pcm",           "Hmm..."),
    "SFX_OH_REALLY":  ("oh_really.pcm",     "Oh, really?"),
    "SFX_WARM_LAUGH": ("warm_laugh.pcm",    "POOL:LAUGH_SOFT"),
}

# ── Filler phrase → cache filename mapping ────────────────────────────────────
# Built at import time from the same filler lists used in language.py.
# Keys are the *raw* filler text (as returned by get_thinking_filler) so we
# can intercept them before the Rime API call.  The values are cache file
# stems, e.g. "filler_default_0" → audio_cache/filler_default_0.pcm.
# IMPORTANT: must stay exactly in sync with _THINKING_FILLERS_BY_CONTEXT in
# language.py — both the phrases AND their order, because PCM files are named
# filler_{ctx}_{index}.pcm.  If you change language.py, update this dict and
# run `python prerecord_audio.py` to regenerate the cached audio files.
_FILLER_PHRASES_BY_CONTEXT = {
    "default": [
        "Gotcha,", "Got it.", "Right,", "I see.",
        "Okay,", "Alright,", "Makes sense.",
    ],
    "friendly": [
        "Perfect,", "Sounds good,", "Great,", "Nice,",
    ],
    "consultative": [
        "Sure,", "Hmm,", "Makes sense,", "Right, okay,",
    ],
    "objection_soft": [
        "I hear you,", "Fair enough,", "Totally get that,",
        "Makes sense,", "Understood,", "Got it,",
    ],
    "gatekeeper": [
        "Of course,", "Totally,", "No problem,", "Got it,", "Sure thing,",
    ],
    "trust_safe": [
        "I get that,", "Totally,", "Makes sense,", "Understood,", "Of course,",
    ],
    "logistics": [
        "No worries,", "All good,", "Sure thing,", "Got it,", "Easy,",
    ],
    "unclear": [
        "Gotcha.", "Right,", "Okay,", "I see.", "Sure,",
    ],
    "playful": [
        "Okay, okay,", "Alright,", "Fair enough,", "Got it,",
    ],
    "reassuring": [
        "Completely understandable.", "I hear that.", "Of course,", "Makes sense,",
    ],
    "surprise": [
        "Oh,", "Hmm.", "Interesting.", "Oh, okay.",
    ],
    "empathy": [
        "I hear you.", "Of course,", "Understood.", "Yeah...",
    ],
    "delight": [
        "Love it.", "Perfect,", "Great,", "Sounds good,",
    ],
    "bridging": [
        "Got it, one sec.", "Okay, quick thought.", "Right, quick one.", "Alright, so,",
    ],
    "thinking": [
        "Umm,", "Hmm,", "Hmm, let me think...",
        "One sec,", "Let me think,", "Umm, let me think...",
    ],
}

# Build the lookup: raw filler text → cache file stem
_FILLER_TEXT_TO_CACHE: dict[str, str] = {}
for _ctx, _phrases in _FILLER_PHRASES_BY_CONTEXT.items():
    for _i, _text in enumerate(_phrases):
        _FILLER_TEXT_TO_CACHE[_text] = f"filler_{_ctx}_{_i}"


# audio_cache/manifest.json maps each recorded stem to the exact text that was
# synthesised, written by prerecord_audio.py. Cached audio is only played when
# the text still matches, so stale recordings can never speak the wrong words.
_MANIFEST_PATH = _AUDIO_CACHE_DIR / "manifest.json"
_manifest_state: dict = {"mtime": None, "data": {}}


def recorded_text(stem: str) -> str | None:
    """Return the text a cached clip was recorded with, or None if unknown."""
    try:
        mtime = _MANIFEST_PATH.stat().st_mtime
    except OSError:
        return None
    if _manifest_state["mtime"] != mtime:
        try:
            import json
            _manifest_state["data"] = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            _manifest_state["mtime"] = mtime
        except Exception as e:
            print(f"[TTS-CACHE] Could not read manifest: {e}")
            return None
    return _manifest_state["data"].get(stem)


def _cache_text_matches(stem: str, expect_text: str | None) -> bool:
    """False only when the manifest proves the recording says something else."""
    if not expect_text:
        return True
    known = recorded_text(stem)
    return known is None or known.strip() == expect_text.strip()


def _lookup_filler_cache(raw_text: str) -> str | None:
    """Return cache file stem if raw_text is a known filler with a pre-recorded PCM, else None."""
    stem = _FILLER_TEXT_TO_CACHE.get(raw_text)
    if stem and (_AUDIO_CACHE_DIR / f"{stem}.pcm").exists() and _cache_text_matches(stem, raw_text):
        return stem
    return None

def _drain_queue(q: asyncio.Queue) -> int:
    """Discard all remaining items in the queue to prevent further TTS API calls."""
    discarded = 0
    while True:
        try:
            q.get_nowait()
            discarded += 1
        except asyncio.QueueEmpty:
            break
    return discarded


def _get_http_client():
    """Lazily initialise the shared async HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        import httpx
        # Per-operation timeouts: generous read timeout for streaming responses,
        # tight connect timeout to fail fast on unreachable servers.
        settings = dict(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
        try:
            _http_client = httpx.AsyncClient(http2=True, **settings)
        except ImportError:
            # HTTP/2 needs the h2 package. Falling back to HTTP/1.1 costs a
            # little latency; refusing to build a client would silence the
            # agent completely, which is far worse on a live call.
            print("[TTS] h2 not installed — using HTTP/1.1 (install httpx[http2] to restore HTTP/2)")
            _http_client = httpx.AsyncClient(http2=False, **settings)
    return _http_client


def _build_rime_payload(text: str, session: Session | None = None) -> dict:
    payload = {
        "text": text,
        "speaker": RIME_VOICE,
        "modelId": RIME_MODEL,
    }
    if RIME_LANG:
        payload["lang"] = RIME_LANG
    if RIME_SAMPLING_RATE:
        payload["samplingRate"] = RIME_SAMPLING_RATE
    # Per-call override (e.g. caller asked NORA to slow down). >1.0 is slower.
    time_scale = getattr(session, "tts_time_scale", None) or RIME_TIME_SCALE_FACTOR
    if time_scale:
        payload["timeScaleFactor"] = time_scale
    if RIME_GENRE:
        payload["genre"] = RIME_GENRE
    return payload


def _strip_wav_header(audio_bytes: bytes) -> bytes:
    if audio_bytes.startswith(b"RIFF"):
        data_start = audio_bytes.find(b"data")
        if data_start != -1:
            return audio_bytes[data_start + 8:]
    return audio_bytes


async def _http_tts_sync(session: Session, text: str, ws_audio_queue: asyncio.Queue) -> int:
    """
    Synchronous fallback: call Rime REST API, wait for full response,
    then stream the decoded PCM bytes to the queue.
    """
    client = _get_http_client()
    total_bytes = 0
    try:
        payload = _build_rime_payload(text, session)
        headers = {
            "Authorization": f"Bearer {RIME_API_KEY}",
            "Content-Type": "application/json",
            "Accept": RIME_ACCEPT_HEADER,
        }
        response = await client.post(_TTS_SYNC_URL, json=payload, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = response.json()
            except Exception:
                data = response.text
            print(f"[Session {session.session_id}] [TTS] Rime returned JSON instead of audio: {data}")
            return 0

        audio_bytes = _strip_wav_header(response.content)

        if session.interrupt_tts.is_set():
            return 0

        for i in range(0, len(audio_bytes), _PCM_CHUNK_SIZE):
            if session.interrupt_tts.is_set():
                break
            chunk = audio_bytes[i:i + _PCM_CHUNK_SIZE]
            total_bytes += len(chunk)
            session.mark_turn("tts_first_audio")
            await ws_audio_queue.put(chunk)
            await asyncio.sleep(0)

    except Exception as e:
        print(f"[Session {session.session_id}] [TTS] HTTP sync error: {e}")
    return total_bytes


async def _http_tts_stream(session: Session, text: str, ws_audio_queue: asyncio.Queue) -> int:
    """
    Stream TTS audio using Rime's endpoint. If the service does not
    support true server streaming, we still stream the buffered bytes
    to the websocket in PCM chunks for low-latency playback.

    Audio chunks are pushed to ws_audio_queue as soon as they arrive from the
    server, achieving sub-200ms time-to-first-audio instead of waiting for the
    full response (~400-1200ms).  Falls back to the sync endpoint automatically
    if the streaming endpoint is unavailable.

    Returns:
        total_bytes: number of PCM bytes streamed to the queue (0 on error/interrupt).
    """
    client = _get_http_client()
    total_bytes = 0
    payload = _build_rime_payload(text, session)
    headers = {
        "Authorization": f"Bearer {RIME_API_KEY}",
        "Content-Type": "application/json",
        "Accept": RIME_ACCEPT_HEADER,
    }

    try:
        async with client.stream(
            "POST", _TTS_STREAM_URL, json=payload, headers=headers,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            # ── JSON response means no audio returned ──
            if "application/json" in content_type:
                try:
                    data = await response.aread()
                    print(
                        f"[Session {session.session_id}] [TTS] Rime returned JSON instead of audio: "
                        f"{data.decode('utf-8', errors='ignore')}"
                    )
                except Exception as e:
                    print(f"[Session {session.session_id}] [TTS] Rime returned JSON instead of audio ({e})")
                return 0

            # ── True binary streaming: raw WAV/PCM chunks ──
            wav_header_stripped = False
            buffer = b""

            async for raw_chunk in response.aiter_bytes(4096):
                if session.interrupt_tts.is_set():
                    break

                buffer += raw_chunk

                # Strip WAV/RIFF header from the first arriving bytes
                if not wav_header_stripped:
                    if buffer.startswith(b"RIFF"):
                        data_marker = buffer.find(b"data")
                        if data_marker == -1:
                            continue  # Need more bytes to find header boundary
                        buffer = buffer[data_marker + 8:]
                    wav_header_stripped = True

                # Push complete PCM chunks to the client immediately
                while len(buffer) >= _PCM_CHUNK_SIZE:
                    if session.interrupt_tts.is_set():
                        break
                    chunk = buffer[:_PCM_CHUNK_SIZE]
                    buffer = buffer[_PCM_CHUNK_SIZE:]
                    total_bytes += len(chunk)
                    session.mark_turn("tts_first_audio")
                    await ws_audio_queue.put(chunk)
                    await asyncio.sleep(0)

            # Flush remaining partial chunk
            if buffer and not session.interrupt_tts.is_set():
                total_bytes += len(buffer)
                await ws_audio_queue.put(buffer)

    except Exception as e:
        if total_bytes == 0:
            # Streaming endpoint failed before any audio — fall back to sync
            print(f"[Session {session.session_id}] [TTS] Stream endpoint failed ({e}), falling back to sync")
            return await _http_tts_sync(session, text, ws_audio_queue)
        print(f"[Session {session.session_id}] [TTS] HTTP streaming error mid-stream: {e}")

    return total_bytes


def _start_buffered_tts(session: Session, text: str) -> tuple[asyncio.Queue, asyncio.Task]:
    """Start synthesising `text` into a buffer right away.

    Used while a cached filler is still playing: Rime streams into the buffer so
    its first bytes arrive during the filler, instead of waiting for the whole
    sentence to be synthesised before anything can play.
    """
    buffer: asyncio.Queue = asyncio.Queue()

    async def _run():
        try:
            await _http_tts_stream(session, text, buffer)
        except Exception as e:
            print(f"[Session {session.session_id}] [TTS] Buffered prefetch error: {e}")
        finally:
            await buffer.put(None)

    return buffer, asyncio.create_task(_run())


async def _drain_buffered_tts(session: Session, buffer: asyncio.Queue, ws_audio_queue: asyncio.Queue) -> int:
    """Forward buffered TTS audio to the client as it becomes available."""
    total = 0
    while True:
        if session.interrupt_tts.is_set():
            break
        chunk = await buffer.get()
        if chunk is None:
            break
        total += len(chunk)
        session.mark_turn("tts_first_audio")
        await ws_audio_queue.put(chunk)
        await asyncio.sleep(0)
    return total


def _split_tts_input_with_sfx(text: str) -> list[tuple[str, str]]:
    """Split text into normal speech and SFX tokens."""
    parts: list[tuple[str, str]] = []
    for chunk in re.split(r"\b(SFX_[A-Z_]+)\b", text):
        if not chunk:
            continue
        normalized = chunk.strip()
        if not normalized:
            continue
        if normalized in _SFX_TOKEN_CONFIG:
            parts.append(("sfx", normalized))
        else:
            parts.append(("text", normalized))
    return parts


async def _stream_cached_sfx(session: Session, token: str, ws_audio_queue: asyncio.Queue) -> int:
    """
    Stream a pre-recorded SFX PCM clip.

    If the PCM file is missing AND a fallback text is configured, the fallback
    is synthesised via the live Rime TTS API so callers always get *some*
    audio for emotional tokens (laughter, sighs, etc.) rather than silence.

    Returns streamed bytes (0 only when both PCM and fallback are unavailable).
    """
    config = _SFX_TOKEN_CONFIG.get(token)
    if not config:
        return 0

    file_name, fallback_text = config
    pcm_path = _AUDIO_CACHE_DIR / file_name

    # Resolve pooled fallback at runtime for varied delivery
    if fallback_text == "POOL:CHUCKLE":
        fallback_text = _pick_laugh_fallback(_LAUGH_CHUCKLE_FALLBACKS)
    elif fallback_text == "POOL:LAUGH_SOFT":
        fallback_text = _pick_laugh_fallback(_LAUGH_SOFT_FALLBACKS)

    # ── Laughter tokens: ALWAYS use live TTS with a varied phrase ──
    # Pre-recorded laugh clips sound robotic because they're the same static
    # recording every time. Live TTS synthesising a full conversational phrase
    # (e.g. "Ha, okay that's fair —") delivers natural warmth and prosodic
    # variation that a single cached clip cannot.
    _LAUGH_TOKENS = {"SFX_CHUCKLE", "SFX_LAUGH_SOFT"}
    if token in _LAUGH_TOKENS and fallback_text and not session.interrupt_tts.is_set():
        print(f"[Session {session.session_id}] [TTS] Laugh via live TTS: '{fallback_text}'")
        return await _http_tts_stream(session, fallback_text, ws_audio_queue)

    # For non-laughter SFX (breaths, sighs): use cached PCM if available
    if not pcm_path.exists():
        # Always fall back to live TTS when a fallback phrase is defined
        if fallback_text and not session.interrupt_tts.is_set():
            print(f"[Session {session.session_id}] [TTS] SFX '{file_name}' not found — live TTS fallback: '{fallback_text}'")
            return await _http_tts_stream(session, fallback_text, ws_audio_queue)
        return 0

    total_bytes = 0
    try:
        with open(pcm_path, "rb") as f:
            while True:
                if session.interrupt_tts.is_set():
                    break
                chunk = f.read(_PCM_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                await ws_audio_queue.put(chunk)
                await asyncio.sleep(0)
    except Exception as e:
        print(f"[Session {session.session_id}] [TTS] SFX error for {file_name}: {e}")
        return 0

    if total_bytes > 0:
        print(f"[Session {session.session_id}] [TTS] SFX: {file_name} ({total_bytes} bytes)")
    return total_bytes


def _maybe_add_breath_token(
    pieces: list[tuple[str, str]],
    idx: int,
    emotion_style: str,
) -> list[tuple[str, str]]:
    """
    Insert a SFX_BREATH_IN token before the next text piece when the emotional
    style calls for it (empathetic / reassuring / resigned).  The breath makes
    the agent sound like it's *composing itself* before addressing something
    sensitive — a key naturalness cue that generic TTS engines miss entirely.

    Only inserts when the *current* piece is a text piece (not back-to-back SFX)
    and a breath hasn't already been inserted just before this position.
    """
    BREATH_STYLES = {"empathetic", "reassuring", "resigned"}
    if emotion_style not in BREATH_STYLES:
        return pieces
    if idx == 0:
        return pieces  # No breath before the very first word
    # Avoid double-breath: skip if previous piece is already SFX_BREATH_IN
    if pieces[idx - 1] == ("sfx", "SFX_BREATH_IN"):
        return pieces
    pieces.insert(idx, ("sfx", "SFX_BREATH_IN"))
    return pieces


async def process_tts_queue(
    session: Session,
    tts_queue: asyncio.Queue,
    ws_audio_queue: asyncio.Queue,
    *,
    primary_intent: str = "UNCLEAR",
    tier: str = "COMMAND",
    refusal_count: int = 0,
):
    """
    Consume text segments from tts_queue, synthesise them via Rime HTTP API,
    and push the resulting PCM audio into ws_audio_queue for the sender task.

    Uses look-ahead pre-fetching: while a chunk is being streamed to the client,
    the next chunk's TTS API call is already in-flight, overlapping network
    latency between consecutive chunks (~200-400ms saved on multi-sentence
    responses).
    """

    async def _prefetch_tts(session: Session, text: str) -> bytes | None:
        """Fire a TTS API call and return raw PCM bytes (or None on error/interrupt)."""
        client = _get_http_client()
        try:
            payload = _build_rime_payload(text, session)
            headers = {
                "Authorization": f"Bearer {RIME_API_KEY}",
                "Content-Type": "application/json",
                "Accept": RIME_ACCEPT_HEADER,
            }
            response = await client.post(
                _TTS_SYNC_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    data = response.json()
                except Exception:
                    data = response.text
                print(f"[Session {session.session_id}] [TTS] Rime returned JSON instead of audio: {data}")
                return None

            audio_bytes = _strip_wav_header(response.content)
            return audio_bytes
        except Exception as e:
            print(f"[Session {session.session_id}] [TTS] Prefetch error: {e}")
            return None

    async def _stream_pcm_bytes(session: Session, audio_bytes: bytes, ws_audio_queue: asyncio.Queue) -> int:
        """Stream raw PCM bytes to the audio queue. Returns total bytes streamed."""
        total = 0
        CHUNK_SIZE = 1600
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            if session.interrupt_tts.is_set():
                break
            chunk = audio_bytes[i:i + CHUNK_SIZE]
            total += len(chunk)
            session.mark_turn("tts_first_audio")
            await ws_audio_queue.put(chunk)
            await asyncio.sleep(0)
        return total

    # Carries a (text, prefetch_task) pair from the filler path to the next
    # segment so the Rime API call and filler playback overlap in time.
    _filler_prefetch: tuple[str, asyncio.Task] | None = None

    # When the filler path consumes the next segment from the queue via a
    # concurrent get(), it stores it here so the next loop iteration skips
    # tts_queue.get() and uses this directly.
    _UNSET = object()
    _pending_segment = _UNSET

    while True:
        if _pending_segment is not _UNSET:
            segment = _pending_segment
            _pending_segment = _UNSET
        else:
            segment = await tts_queue.get()

        if segment is None:
            tts_queue.task_done()
            break

        # ── Interrupt check BEFORE making any API call ──
        if session.interrupt_tts.is_set():
            tts_queue.task_done()
            if _filler_prefetch:
                _filler_prefetch[2].cancel()
                _filler_prefetch = None
            discarded = _drain_queue(tts_queue)
            if discarded:
                print(f"[Session {session.session_id}] [TTS] Interrupted — skipped {discarded} queued segment(s).")
            break

        # ── Filler cache fast-path ──
        # If this segment is a known filler phrase with a pre-recorded PCM,
        # stream it from disk (~0ms) instead of calling the Rime API (~400-1200ms).
        # This is the biggest single latency win for most turns.
        filler_cache_stem = _lookup_filler_cache(segment)
        if filler_cache_stem:
            pcm_path = _AUDIO_CACHE_DIR / f"{filler_cache_stem}.pcm"
            await session.set_state(AssistantState.SPEAKING)
            filler_bytes = 0

            # ── Concurrent prefetch: wait for the next segment NOW ──────────
            # Start an async tts_queue.get() that runs IN PARALLEL with filler
            # playback.  The old peek-based approach always failed because the
            # queue was empty (LLM hadn't generated anything yet).  This new
            # approach properly blocks until the LLM pushes its first chunk,
            # then immediately fires the TTS API call — all while the cached
            # filler audio is still playing.
            _next_seg_task = (
                asyncio.create_task(tts_queue.get())
                if _filler_prefetch is None
                   and RIME_API_KEY
                   and not session.interrupt_tts.is_set()
                else None
            )

            try:
                with open(pcm_path, "rb") as f:
                    while True:
                        if session.interrupt_tts.is_set():
                            break
                        chunk = f.read(_PCM_CHUNK_SIZE)
                        if not chunk:
                            break
                        filler_bytes += len(chunk)
                        session.mark_turn("filler_audio")
                        await ws_audio_queue.put(chunk)
                        await asyncio.sleep(0)

                        # ── Mid-filler prefetch trigger ──────────────────────
                        # If the LLM pushed a chunk while the filler is still
                        # playing, start the TTS API call RIGHT NOW so it
                        # overlaps with the remaining filler audio.
                        if (
                            _next_seg_task is not None
                            and _next_seg_task.done()
                            and _filler_prefetch is None
                            and not session.interrupt_tts.is_set()
                        ):
                            _next_raw = _next_seg_task.result()
                            if _next_raw is not None and not _lookup_filler_cache(_next_raw):
                                # Pre-process identically to the main path so
                                # _filler_prefetch[0] matches piece_value later.
                                _pre = enhance_tts_text(
                                    _next_raw,
                                    primary_intent=primary_intent,
                                    tier=tier,
                                    refusal_count=refusal_count,
                                )
                                _pre_chunks = split_response_for_tts(_pre)
                                if _pre_chunks:
                                    _pre_pieces = []
                                    for _tc in _pre_chunks:
                                        _pre_pieces.extend(_split_tts_input_with_sfx(_tc))
                                    for _pt, _pv in _pre_pieces:
                                        if _pt == "text":
                                            _buf, _task = _start_buffered_tts(session, _pv)
                                            _filler_prefetch = (_pv, _buf, _task)
                                            break
            except Exception as e:
                print(f"[Session {session.session_id}] [TTS] Filler cache error: {e}")

            if filler_bytes > 0 and not session.interrupt_tts.is_set():
                duration = (filler_bytes / 32000.0) + 0.3
                await session.add_speaking_duration(duration)
                print(f"[Session {session.session_id}] [TTS] Filler from cache: {filler_cache_stem} ({filler_bytes} bytes)")

            tts_queue.task_done()

            # ── Resolve the concurrent segment wait ──────────────────────────
            if session.interrupt_tts.is_set():
                if _next_seg_task and not _next_seg_task.done():
                    _next_seg_task.cancel()
                if _filler_prefetch:
                    _filler_prefetch[2].cancel()
                    _filler_prefetch = None
                discarded = _drain_queue(tts_queue)
                if discarded:
                    print(f"[Session {session.session_id}] [TTS] Interrupted during filler — skipped {discarded} queued segment(s).")
                break

            if _next_seg_task is not None:
                if not _next_seg_task.done():
                    # LLM still generating — await the segment, then start prefetch
                    _pending_segment = await _next_seg_task
                    if (
                        _pending_segment is not None
                        and not _lookup_filler_cache(_pending_segment)
                        and _filler_prefetch is None
                        and not session.interrupt_tts.is_set()
                    ):
                        _pre = enhance_tts_text(
                            _pending_segment,
                            primary_intent=primary_intent,
                            tier=tier,
                            refusal_count=refusal_count,
                        )
                        _pre_chunks = split_response_for_tts(_pre)
                        if _pre_chunks:
                            _pre_pieces = []
                            for _tc in _pre_chunks:
                                _pre_pieces.extend(_split_tts_input_with_sfx(_tc))
                            for _pt, _pv in _pre_pieces:
                                if _pt == "text":
                                    _buf, _task = _start_buffered_tts(session, _pv)
                                    _filler_prefetch = (_pv, _buf, _task)
                                    break
                else:
                    _pending_segment = _next_seg_task.result()
                    # Prefetch was already started mid-filler if possible
            continue

        # ── Laugh-opener preprocessing ──────────────────────────────────────────
        # If the LLM starts a segment with a bare laugh syllable ("Ha, …"),
        # replace it with SFX_CHUCKLE so the varied-phrase pool plays a natural-
        # sounding laugh via live TTS, then the rest of the sentence follows
        # cleanly.  Handles "Ha,", "Haha!", "Hah." etc.
        ha_match = _HA_LAUGH_RE.match(segment)
        if ha_match:
            rest = segment[ha_match.end():].strip()
            if rest:
                segment = f"SFX_CHUCKLE {rest}"
            else:
                segment = "SFX_CHUCKLE"

        processed = enhance_tts_text(
            segment,
            primary_intent=primary_intent,
            tier=tier,
            refusal_count=refusal_count,
        )
        tts_chunks = split_response_for_tts(processed)

        if not tts_chunks:
            tts_queue.task_done()
            continue

        if not RIME_API_KEY:
            print(f"[Session {session.session_id}] [TTS] No Rime API key.")
            tts_queue.task_done()
            continue

        await session.set_state(AssistantState.SPEAKING)

        interrupted_mid_stream = False
        try:
            t_start = time.time()
            total_bytes = 0

            # Flatten all chunks into individual text+sfx pieces for pre-fetching
            all_pieces: list[tuple[str, str]] = []
            for tts_input in tts_chunks:
                all_pieces.extend(_split_tts_input_with_sfx(tts_input))

            # Inject a breath token before the first text piece on sensitive emotional
            # turns so the agent sounds like it’s composing itself before speaking.
            _emotion_style = _resolve_emotion_style(
                primary_intent=primary_intent,
                tier=tier,
                refusal_count=refusal_count,
            )
            if _emotion_style in {"empathetic", "reassuring", "resigned"}:
                # Find the first text piece and insert a breath just before it
                for _bi, (_bt, _bv) in enumerate(all_pieces):
                    if _bt == "text":
                        all_pieces = _maybe_add_breath_token(all_pieces, _bi, _emotion_style)
                        break

            # Process pieces with one-step look-ahead pre-fetch
            prefetch_task: asyncio.Task | None = None

            for idx, (piece_type, piece_value) in enumerate(all_pieces):
                if session.interrupt_tts.is_set():
                    interrupted_mid_stream = True
                    break

                if piece_type == "sfx":
                    # _stream_cached_sfx now handles the live-TTS fallback internally
                    # when the PCM file is absent, so no manual fallback needed here.
                    bytes_streamed = await _stream_cached_sfx(session, piece_value, ws_audio_queue)
                    total_bytes += bytes_streamed
                else:
                    print(f"[Session {session.session_id}] [TTS] Generating: {piece_value}")

                    # ── Use filler-time prefetch if available ──────────────────
                    # Synthesis started while the filler played; stream whatever has
                    # arrived so far and keep streaming the rest.
                    if (
                        _filler_prefetch is not None
                        and _filler_prefetch[0] == piece_value
                        and prefetch_task is None
                    ):
                        _, _buffer, _ = _filler_prefetch
                        _filler_prefetch = None
                        total_bytes += await _drain_buffered_tts(session, _buffer, ws_audio_queue)
                        # Start the NEXT text piece while this one plays out.
                        next_text_piece = None
                        for future_type, future_value in all_pieces[idx + 1:]:
                            if future_type == "text":
                                next_text_piece = future_value
                                break
                        if next_text_piece and not session.interrupt_tts.is_set():
                            prefetch_task = asyncio.create_task(_prefetch_tts(session, next_text_piece))
                        if session.interrupt_tts.is_set():
                            interrupted_mid_stream = True
                            break
                        continue

                    # If we pre-fetched this chunk, await the result instead of a new call
                    if prefetch_task is not None:
                        prefetch_result = await prefetch_task
                        prefetch_task = None
                    else:
                        # No prefetch available — use streaming endpoint directly
                        # for lowest time-to-first-audio. Start prefetching the NEXT
                        # text piece in parallel so it's ready when this one finishes.
                        next_text_piece = None
                        for future_type, future_value in all_pieces[idx + 1:]:
                            if future_type == "text":
                                next_text_piece = future_value
                                break
                        if next_text_piece and not session.interrupt_tts.is_set():
                            prefetch_task = asyncio.create_task(_prefetch_tts(session, next_text_piece))

                        # Stream current piece directly — audio chunks arrive as
                        # Rime generates them (sub-200ms first-chunk latency when supported)
                        bytes_streamed = await _http_tts_stream(session, piece_value, ws_audio_queue)
                        total_bytes += bytes_streamed
                        prefetch_result = None  # already streamed
                        continue  # skip the prefetch_result streaming below

                    # Start pre-fetching the NEXT text piece while we stream this one
                    # (only if we didn't already start a prefetch above)
                    if prefetch_task is None:
                        next_text_piece = None
                        for future_type, future_value in all_pieces[idx + 1:]:
                            if future_type == "text":
                                next_text_piece = future_value
                                break
                        if next_text_piece and not session.interrupt_tts.is_set():
                            prefetch_task = asyncio.create_task(_prefetch_tts(session, next_text_piece))

                    # Stream the prefetched result
                    if prefetch_result and not session.interrupt_tts.is_set():
                        bytes_streamed = await _stream_pcm_bytes(session, prefetch_result, ws_audio_queue)
                        total_bytes += bytes_streamed
                    elif not prefetch_result:
                        # Prefetch failed — stream directly via streaming endpoint
                        bytes_streamed = await _http_tts_stream(session, piece_value, ws_audio_queue)
                        total_bytes += bytes_streamed

                if session.interrupt_tts.is_set():
                    interrupted_mid_stream = True
                    break

            # Cancel any outstanding prefetch tasks
            for _task in filter(None, [prefetch_task, _filler_prefetch[2] if _filler_prefetch else None]):
                if not _task.done():
                    _task.cancel()
                    try:
                        await _task
                    except (asyncio.CancelledError, Exception):
                        pass
            if _filler_prefetch:
                _filler_prefetch = None

            t_elapsed = time.time() - t_start
            print(f"[Session {session.session_id}] [TTS] Streamed {total_bytes} bytes in {t_elapsed:.2f}s (HTTP)")

            # Calculate duration for playback tracking
            # 16kHz, 16-bit (2 bytes), mono = 32,000 bytes per second
            if not interrupted_mid_stream and not session.interrupt_tts.is_set() and total_bytes > 0:
                duration = (total_bytes / 32000.0) + 0.5
                await session.add_speaking_duration(duration)

        except Exception as e:
            print(f"[Session {session.session_id}] [TTS] Error: {e}")

        finally:
            tts_queue.task_done()

        # ── Interrupt check AFTER the API call ──
        if interrupted_mid_stream:
            discarded = _drain_queue(tts_queue)
            if discarded:
                print(f"[Session {session.session_id}] [TTS] Skipped {discarded} queued segment(s) after interruption.")
            break

    print(f"[Session {session.session_id}] [TTS] Queue processed. Waiting for physical playback...")

    # Wait for the calculated end of physical audio playback
    now = time.time()
    while now < session.expected_speech_end_time and not session.interrupt_tts.is_set():
        await asyncio.sleep(0.1)
        now = time.time()

    # Restore state to LISTENING when TTS queue is fully done (and not interrupted)
    if await session.get_state() == AssistantState.SPEAKING:
        await session.set_state(AssistantState.LISTENING)


async def speak_text(
    session: Session,
    text: str,
    ws_audio_queue: asyncio.Queue,
    *,
    allow_immediate_barge_in: bool = False,
) -> None:
    """
    Speak a single text string using the live TTS path.

    When allow_immediate_barge_in is True, interruption detection will not
    enforce the echo grace period for this segment.
    """
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait(text)
    q.put_nowait(None)

    previous = getattr(session, "allow_barge_in_immediate", False)
    session.allow_barge_in_immediate = bool(allow_immediate_barge_in)
    try:
        await process_tts_queue(session, q, ws_audio_queue)
    finally:
        session.allow_barge_in_immediate = previous


# ── Pre-recorded audio cache ────────────────────────────────────────────────
AUDIO_CACHE_DIR = _AUDIO_CACHE_DIR


async def play_cached_audio(
    session: "Session",
    name: str,
    ws_audio_queue: asyncio.Queue,
    fallback_text: str = "",
    expect_text: str = "",
) -> None:
    """
    Stream a pre-recorded PCM file from audio_cache/<name>.pcm to the client.

    Manages session state (SPEAKING -> LISTENING) and expected_speech_end_time
    in exactly the same way as process_tts_queue so the rest of the pipeline
    (VAD echo-grace, interruption logic, silence timeout) behaves identically.

    Args:
        session:         Active Session object.
        name:            File stem, e.g. "greeting" -> audio_cache/greeting.pcm
        ws_audio_queue:  Queue that the sender task reads from.
        fallback_text:   If the .pcm file is missing, synthesise this text instead
                         via the live Rime pipeline.  Pass "" to skip.
    """
    pcm_path = AUDIO_CACHE_DIR / f"{name}.pcm"

    # A recording made for different text (e.g. another customer's name) must
    # never be played; fall back to live TTS of the text we actually want.
    if pcm_path.exists() and not _cache_text_matches(name, expect_text or fallback_text):
        print(f"[TTS-CACHE] '{name}.pcm' was recorded for different text — using live TTS.")
        pcm_path = AUDIO_CACHE_DIR / "__mismatched__.pcm"

    # Always clear stale interrupt flags — this is a deliberate terminal playback
    # (greeting, farewell, timeout). A leftover interrupt_tts from a previous turn
    # would otherwise cause the file to be silently skipped.
    session.interrupt_tts.clear()

    if not pcm_path.exists():
        print(f"[TTS-CACHE] '{name}.pcm' not found — {'falling back to live TTS' if fallback_text else 'skipping'}.")
        if fallback_text:
            q: asyncio.Queue = asyncio.Queue()
            q.put_nowait(fallback_text)
            q.put_nowait(None)
            await process_tts_queue(session, q, ws_audio_queue)
        return

    # ── Early interrupt check ─────────────────────────────────────────────────
    if session.interrupt_tts.is_set():
        print(f"[TTS-CACHE] Skipping '{name}.pcm' — interrupted before playback.")
        return

    await session.set_state(AssistantState.SPEAKING)

    total_bytes = 0
    interrupted = False

    try:
        with open(pcm_path, "rb") as f:
            while True:
                chunk = f.read(_PCM_CHUNK_SIZE)
                if not chunk:
                    break  # EOF

                if session.interrupt_tts.is_set():
                    print(f"[TTS-CACHE] '{name}.pcm' interrupted mid-playback.")
                    interrupted = True
                    break

                total_bytes += len(chunk)
                await ws_audio_queue.put(chunk)
                # Yield control so the sender task can actually transmit the chunk
                await asyncio.sleep(0)

    except Exception as e:
        print(f"[TTS-CACHE] Error reading '{name}.pcm': {e}")
        interrupted = True

    if not interrupted and total_bytes > 0:
        # 16-bit PCM @ 16 kHz mono = 32 000 bytes/sec; add 0.5 s network buffer
        duration = (total_bytes / 32_000.0) + 0.5
        await session.add_speaking_duration(duration)

        print(f"[TTS-CACHE] Streamed '{name}.pcm' ({total_bytes / 1024:.1f} KB, {duration - 0.5:.2f}s)")

        # Wait for physical playback to finish before releasing LISTENING state
        now = time.time()
        while now < session.expected_speech_end_time and not session.interrupt_tts.is_set():
            await asyncio.sleep(0.1)
            now = time.time()

    if await session.get_state() == AssistantState.SPEAKING:
        await session.set_state(AssistantState.LISTENING)


async def warm_up_tts_connection() -> None:
    """Open the pooled HTTP/2 connection to Rime before the first real request.

    Any response (even 404) completes DNS + TLS, which is the expensive part.
    """
    if not RIME_API_KEY:
        return
    try:
        client = _get_http_client()
        await client.get("https://users.rime.ai/", timeout=5.0)
    except Exception:
        pass


async def close_tts_connection(session: Session):
    """
    Clean up TTS resources for a session.
    (No-op now that we use HTTP-only — kept for API compatibility with main.py.)
    """
    pass