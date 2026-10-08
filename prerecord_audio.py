"""
prerecord_audio.py
==================
Standalone script to pre-record all static/hardcoded voice messages used by
the Awaaz voice assistant pipeline.

Run once (or whenever messages change) to generate raw PCM audio files:

    python prerecord_audio.py

Files are saved to ./audio_cache/ as 16-bit signed PCM, 16 kHz, mono —
byte-perfect with the existing Rime pipeline output format.

Re-running is safe: existing files are skipped unless you pass --force.
"""

import asyncio
import json
import os
import sys
import argparse

import websockets
from dotenv import load_dotenv

# ── Load credentials from .env (same as the main app) ─────────────────────────
load_dotenv()
RIME_API_KEY = os.getenv("RIME_API_KEY")


def _get_rime_time_scale_factor() -> float:
    raw = os.getenv("RIME_TIME_SCALE_FACTOR", os.getenv("RIME_SPEED_ALPHA", "1.0")).strip()
    try:
        factor = float(raw)
    except ValueError:
        return 1.0
    return max(0.6, min(1.6, factor))


def _get_rime_repetition_penalty() -> float:
    raw = os.getenv("RIME_REPETITION_PENALTY", "1.1").strip()
    try:
        penalty = float(raw)
    except ValueError:
        return 1.1
    return max(0.8, min(2.0, penalty))


def _get_rime_sampling_rate() -> int:
    raw = os.getenv("RIME_SAMPLING_RATE", "16000").strip()
    try:
        rate = int(raw)
    except ValueError:
        return 16000
    return 16000 if rate <= 0 else rate

# ── Must match the live pipeline settings in tts.py ───────────────────────────
RIME_VOICE = os.getenv("RIME_VOICE", "orion").strip()
RIME_MODEL = os.getenv("RIME_MODEL", "arcanav2").strip()
RIME_GENRE = os.getenv("RIME_GENRE", "conversational").strip()
RIME_AUDIO_FORMAT = os.getenv("RIME_AUDIO_FORMAT", "wav").strip().lower()
RIME_LANG = os.getenv("RIME_LANG", "").strip()
RIME_SAMPLING_RATE = _get_rime_sampling_rate()
RIME_TIME_SCALE_FACTOR = _get_rime_time_scale_factor()
RIME_REPETITION_PENALTY = _get_rime_repetition_penalty()
RIME_ACCEPT_HEADER = (
    RIME_AUDIO_FORMAT if RIME_AUDIO_FORMAT.startswith("audio/") else {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mp3",
        "opus": "audio/webm;codecs=opus",
        "ogg": "audio/ogg;codecs=opus",
        "mulaw": "audio/x-mulaw",
    }.get(RIME_AUDIO_FORMAT, "audio/wav")
)

# ── Output directory (created next to this script) ────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")

# ── Static messages to pre-record ─────────────────────────────────────────────
# Format: { "output_filename_without_extension": "exact message text" }
# These must match the strings used in main.py exactly.

# ── Greeting variants (pre-recorded to eliminate TTS cold-start latency) ──────
# The {name} placeholder is replaced by the sample lead name at record time.
# main.py selects the matching .pcm file instead of calling TTS live.
_SAMPLE_LEAD_NAME = "Ehsan Khan"
_SAMPLE_SPECIALTY = "medical"
_SAMPLE_CITY = "Dallas"

from app.services.policy import cacheable_templates
from app.session import SAMPLE_LEAD_CONTEXT
from app.utils.language import build_greeting_variants

# Greetings are recorded with the sample lead/organisation. For any other lead
# the manifest check in tts.py makes the agent fall back to live TTS, so a
# recording can never speak the wrong name.
_SAMPLE_LEAD_NAME = SAMPLE_LEAD_CONTEXT.get("lead_name", "")
_SAMPLE_PRACTICE = SAMPLE_LEAD_CONTEXT.get("practice_name", "the practice")

MESSAGES: dict[str, str] = {
    # ── Session lifecycle ──────────────────────────────────────────────────────
    "reconnected":      "Reconnected.",

    # ── Timeout / limit farewells ──────────────────────────────────────────────
    "inactivity_bye":   "It seems like you've stepped away. Thanks for your time. Goodbye!",
    "timeout_bye":      "I don't want to take up too much of your time. Thanks for chatting, have a great day!",
    "turn_limit_bye":   "I appreciate your time. I'll let you go. Have a great day!",

    # ── User-initiated exit ────────────────────────────────────────────────────
    "decline_bye":      "Understood. I will not keep you. Thanks for your time, and have a good day.",

    # ── Emotional SFX clips ────────────────────────────────────────────────────
    # These are used by the SFX token system in tts.py (SFX_CHUCKLE, SFX_LAUGH_SOFT,
    # SFX_SIGH_SOFT) to play pre-recorded natural-sounding cues instantly — no API
    # round-trip at call time.
    #
    # Phrasing strategy: full contextual sentences rather than bare phonetics
    # ("heh") so Rime's expressive model delivers them with natural warmth,
    # rising prosody, and a genuine laugh quality rather than a flat literal read.
    # Note: the live system now prefers real-time TTS with random phrase selection
    # over these static clips for better variety. These serve as backup only.
    "laugh_chuckle":    "Ha, okay — I like that.",
    "laugh_soft":       "Ha, honestly, yeah, that's fair.",
    "sigh_soft":        "Mm... yeah, I hear you.",

    # Breath token — a brief natural pause/inhale before sensitive turns.
    # A very short silence filler; the phrase is intentionally minimal so
    # Rime renders just the breath onset, not a full sentence.
    "breath_in":        "Mm.",
}

# ── Add all greeting variants to the pre-record list ──────────────────────────
for _name, _text in build_greeting_variants(_SAMPLE_LEAD_NAME, _SAMPLE_PRACTICE):
    MESSAGES[_name] = _text
for _name, _text in build_greeting_variants("", _SAMPLE_PRACTICE):
    MESSAGES[_name] = _text

# ── Fixed policy lines (do-not-call, hold, goodbyes, escalation offers) ───────
# Playing these from disk removes a 1-2 s Rime call from every policy turn.
for _name, _text in cacheable_templates().items():
    MESSAGES[f"tpl_{_name}"] = _text

# ── Add all thinking filler phrases so they play from cache (~0ms) ────────────
# These are the exact strings from app/utils/language.py's _THINKING_FILLERS_BY_CONTEXT.
# At runtime, tts.py checks for a cached PCM before calling the Rime API.
# Key format: filler_<context>_<index> to keep filenames deterministic.
# IMPORTANT: must stay exactly in sync with _THINKING_FILLERS_BY_CONTEXT in
# language.py AND _FILLER_PHRASES_BY_CONTEXT in tts.py — same phrases, same order.
# After any change here, re-run: python prerecord_audio.py
_FILLER_PHRASES = {
    "default": [
        "Gotcha,",
        "Got it.",
        "Right,",
        "I see.",
        "Okay,",
        "Alright,",
        "Makes sense.",
    ],
    "friendly": [
        "Perfect,",
        "Sounds good,",
        "Great,",
        "Nice,",
    ],
    "consultative": [
        "Sure,",
        "Hmm,",
        "Makes sense,",
        "Right, okay,",
    ],
    "objection_soft": [
        "I hear you,",
        "Fair enough,",
        "Totally get that,",
        "Makes sense,",
        "Understood,",
        "Got it,",
    ],
    "gatekeeper": [
        "Of course,",
        "Totally,",
        "No problem,",
        "Got it,",
        "Sure thing,",
    ],
    "trust_safe": [
        "I get that,",
        "Totally,",
        "Makes sense,",
        "Understood,",
        "Of course,",
    ],
    "logistics": [
        "No worries,",
        "All good,",
        "Sure thing,",
        "Got it,",
        "Easy,",
    ],
    "unclear": [
        "Gotcha.",
        "Right,",
        "Okay,",
        "I see.",
        "Sure,",
    ],
    "playful": [
        "Okay, okay,",
        "Alright,",
        "Fair enough,",
        "Got it,",
    ],
    "reassuring": [
        "Completely understandable.",
        "I hear that.",
        "Of course,",
        "Makes sense,",
    ],
    "surprise": [
        "Oh,",
        "Hmm.",
        "Interesting.",
        "Oh, okay.",
    ],
    "empathy": [
        "I hear you.",
        "Of course,",
        "Understood.",
        "Yeah...",
    ],
    "delight": [
        "Love it.",
        "Perfect,",
        "Great,",
        "Sounds good,",
    ],
    "bridging": [
        "Got it, one sec.",
        "Okay, quick thought.",
        "Right, quick one.",
        "Alright, so,",
    ],
    # ── Genuine thinking pause sounds ─────────────────────────────────────────
    # Must match language.py and tts.py exactly (same order).
    "thinking": [
        "Umm,",
        "Hmm,",
        "Hmm, let me think...",
        "One sec,",
        "Let me think,",
        "Umm, let me think...",
    ],
}
for ctx, phrases in _FILLER_PHRASES.items():
    for i, text in enumerate(phrases):
        MESSAGES[f"filler_{ctx}_{i}"] = text


async def generate_audio(text: str) -> bytes:
    """
    Call Rime HTTP REST API and return all PCM bytes for `text`.
    """
    import httpx
    url = "https://users.rime.ai/v1/rime-tts"
    payload = {
        "text": text,
        "speaker": RIME_VOICE,
        "modelId": RIME_MODEL,
        "samplingRate": RIME_SAMPLING_RATE,
        "timeScaleFactor": RIME_TIME_SCALE_FACTOR,
    }
    if RIME_LANG:
        payload["lang"] = RIME_LANG
    if RIME_GENRE:
        payload["genre"] = RIME_GENRE
    headers = {
        "Authorization": f"Bearer {RIME_API_KEY}",
        "Content-Type": "application/json",
        "Accept": RIME_ACCEPT_HEADER,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            print("  [WARN] Rime returned JSON instead of audio: " + response.text)
            return b""

        audio_bytes = response.content

        # Strip WAV header if present
        if audio_bytes.startswith(b"RIFF"):
            data_start = audio_bytes.find(b"data")
            if data_start != -1:
                audio_bytes = audio_bytes[data_start + 8:]

        return audio_bytes


async def main(force: bool = False) -> None:
    if not RIME_API_KEY:
        print("[ERROR] RIME_API_KEY not found in .env. Aborting.")
        sys.exit(1)

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[INFO] Audio cache directory: {CACHE_DIR}")
    print(
        f"[INFO] Voice: {RIME_VOICE}  |  Model: {RIME_MODEL}  |  Genre: {RIME_GENRE}  "
        f"|  Format: {RIME_AUDIO_FORMAT}  |  TimeScale: {RIME_TIME_SCALE_FACTOR}"
    )
    print()

    # manifest.json records the exact text behind every clip so the agent can
    # detect stale recordings at call time (see tts.py recorded_text()).
    manifest_path = os.path.join(CACHE_DIR, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            _manifest = json.load(f)
    except Exception:
        _manifest = {}
    _new_manifest: dict[str, str] = {}

    total = len(MESSAGES)
    generated = 0
    skipped = 0

    for name, text in MESSAGES.items():
        out_path = os.path.join(CACHE_DIR, f"{name}.pcm")

        if os.path.exists(out_path) and not force and _manifest.get(name) == text:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  [SKIP] {name}.pcm  ({size_kb:.1f} KB) — already matches. Use --force to re-generate.")
            skipped += 1
            _new_manifest[name] = text
            continue
        if os.path.exists(out_path) and _manifest.get(name) != text:
            print(f"  [STALE] {name}.pcm was recorded for different text — re-generating.")

        print(f"  [GEN]  {name}.pcm  <- \"{text}\"")
        try:
            audio_bytes = await generate_audio(text)

            if not audio_bytes:
                print(f"  [WARN] {name}.pcm — Rime returned empty audio. Skipping save.")
                continue

            with open(out_path, "wb") as f:
                f.write(audio_bytes)

            _new_manifest[name] = text
            size_kb = len(audio_bytes) / 1024
            # 16-bit PCM @ 16 kHz → 32,000 bytes/sec
            duration_s = len(audio_bytes) / 32_000
            print(f"         Saved {size_kb:.1f} KB  ({duration_s:.2f}s of audio)")
            generated += 1

        except Exception as e:
            print(f"  [ERROR] Failed to generate {name}.pcm: {e}")

    print()
    print(f"[DONE] {generated} generated, {skipped} skipped out of {total} total messages.")
    if generated + skipped < total:
        print(f"       {total - generated - skipped} failed — check errors above.")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_new_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[INFO] Wrote manifest with {len(_new_manifest)} entries -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-record static Awaaz voice messages via Rime TTS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate all files even if they already exist.",
    )
    args = parser.parse_args()

    asyncio.run(main(force=args.force))
