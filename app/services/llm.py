import asyncio
import re
from groq import AsyncGroq
import os
from dotenv import load_dotenv

from app.session import Session
from app.services.policy import template
from app.utils.language import build_system_prompt

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Override with GROQ_REPLY_MODEL when Groq retires a model.
# Note: openai/gpt-oss-* models consume max_tokens on hidden reasoning and can
# return no spoken content at these limits — raise max_tokens before using one.
REPLY_MODEL = os.getenv("GROQ_REPLY_MODEL", "qwen/qwen3.8-27b").strip()

_CLOSING_PATTERNS = [
    r"thank you for your help",
    r"thanks for your help",
    r"have a great day",
    r"have a nice day",
    r"goodbye",
    r"good bye",
]


def _response_has_closing(response: str) -> bool:
    lowered = (response or "").lower()
    return any(re.search(p, lowered) for p in _CLOSING_PATTERNS)


# Spoken when a wrap-up goodbye can't be generated.
_WRAP_UP_FALLBACK = "Thank you so much for your time. Have a great day!"


async def _handle_generation_failure(session: Session, tts_queue: asyncio.Queue, primary_intent: str, spoke: bool) -> None:
    """Never leave the caller in silence when generation fails.

    - Wrap-up turns still say goodbye (and the call closes after it).
    - The first failure asks the caller to repeat themselves.
    - A second consecutive failure says nothing here; main.py sees
      session.llm_failures >= 2 and ends the call with a technical-trouble goodbye.
    """
    session.llm_failures += 1
    if not spoke:
        if primary_intent == "WRAP_UP_TIMEOUT":
            await tts_queue.put(_WRAP_UP_FALLBACK)
            session.close_after_speaking = True
        elif session.llm_failures < 2:
            await tts_queue.put(template("LLM_RETRY"))
    await tts_queue.put(None)


async def warm_up_llm_connection() -> None:
    """Open this client's HTTPS connection early so turn 1 skips the TLS handshake.

    Uses a models listing (no tokens generated). Failures are ignored.
    """
    try:
        await client.models.list()
    except Exception:
        pass


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _has_spoken_content(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))

async def get_ai_response_stream(
    user_input: str,
    lang: str,
    session: Session,
    tts_queue: asyncio.Queue,
    max_tokens: int = 100,
    max_words: int = 40,
    tier: str = "COMMAND",
    primary_intent: str = "UNCLEAR",
    filler_used: bool = False,
    current_mode: str = "INTRO",
    recent_openers: list = None,
    pitch_delivered: bool = False,
):
    """
    Streams output from Groq, splits it into sentences, and pushes them to tts_queue.
    max_tokens and max_words are set dynamically by the intent classifier.
    """
    system_prompt = build_system_prompt(
        lang,
        max_words=max_words,
        tier=tier,
        primary_intent=primary_intent,
        refusal_count=session.refusal_count,
        meeting_ask_count=session.meeting_ask_count,
        lead_context=session.lead_context,
        last_objection=session.last_objection,
        turn_count=session.turn_count,
        ai_ask_count=session.ai_ask_count,
        filler_used=filler_used,
        current_mode=current_mode,
        recent_openers=recent_openers,
        email_captured=session.email_captured,
        email_address=session.email_address,
        email_pending_confirmation=session.email_pending_confirmation,
        mood_trajectory=session.mood_trajectory,
        pitch_delivered=pitch_delivered,
        name_clarify_attempts=session.name_clarify_attempts,
        stage=getattr(session, "call_stage", ""),
        identity_verified=getattr(session, "identity_verified", False),
        reminder_delivered=getattr(session, "reminder_delivered", False),
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session.chat_history[-12:])
    messages.append({"role": "user", "content": user_input})

    full_response = ""
    buffer = ""
    pending_tts_fragment = ""
    emitted_chunks = 0
    stream_interrupted = False

    try:
        stream = await client.chat.completions.create(
            model=REPLY_MODEL,
            messages=messages,
            temperature=0.82,  # Higher variation → more natural, less scripted phrasing
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if session.interrupt_llm.is_set():
                stream_interrupted = True
                print(f"[Session {session.session_id}] [LLM] Stream cancelled by interruption.")
                break

            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            
            if not content:
                continue

            if not full_response:
                session.mark_turn("llm_first_token")
            full_response += content
            buffer += content

            if not stream_interrupted:
                while True:
                    # Flush on punctuation boundaries.  We match all
                    # punctuation first, then skip periods that follow
                    # common abbreviations (Dr., Mr., etc.) so they
                    # don't create tiny 1-word TTS fragments.
                    match = re.search(r"[,.!?;:\n]", buffer)
                    if not match:
                        break  # Wait for more tokens until we hit punctuation

                    # Skip comma splits in thousands separators ("10,000").
                    if match.group() == ",":
                        pre = buffer[:match.start()]
                        post = buffer[match.end():]
                        if re.search(r"\d$", pre):
                            if re.match(r"\s*\d{0,2}$", post):
                                break
                            if re.match(r"\s*\d{3}\b", post):
                                buffer = pre + post.lstrip()
                                continue

                    # Skip abbreviation periods — "Dr." "Mr." etc.
                    if match.group() == ".":
                        pre = buffer[:match.start()]
                        post_char = buffer[match.end():match.end() + 1]

                        # Email dots are not sentence boundaries. Convert them to
                        # spoken 'dot' to keep TTS natural and prevent extra chunks.
                        email_window = buffer[max(0, match.start() - 48):min(len(buffer), match.end() + 48)]
                        if (
                            pre
                            and re.search(r"[A-Za-z0-9]$", pre)
                            and post_char
                            and re.search(r"[A-Za-z0-9]", post_char)
                            and "@" in email_window
                        ):
                            buffer = buffer[:match.start()] + " dot " + buffer[match.end():]
                            continue

                        if re.search(r"\b(?:Dr|Mr|Ms|Mrs|St|vs|Jr|Sr|etc|Inc)$", pre):
                            # Not a real sentence boundary — skip past it
                            # and keep scanning for the next punctuation.
                            buffer = buffer[:match.start()] + buffer[match.start()+1:]
                            continue
                    
                    end_idx = match.end()
                    candidate = buffer[:end_idx].strip()
                    buffer = buffer[end_idx:].lstrip()
                    if not candidate:
                        continue

                    if not _has_spoken_content(candidate):
                        continue

                    words = _word_count(candidate)
                    is_terminal = bool(re.search(r"[.!?;]\s*$", candidate))
                    is_pause = bool(re.search(r"[,:]\s*$", candidate))

                    # Prevent tiny opener fragments like "So look,." or "Okay,."
                    # from becoming standalone TTS calls. Pause-delimited chunks
                    # (comma/colon) need much more context to sound natural.
                    # First chunk uses a lower threshold (2 words) to get audio
                    # to TTS faster and reduce the gap after filler playback.
                    # After the opening chunk, short fragments ("Dr Mitchell.")
                    # are held back and merged into the next one: sent alone they
                    # cost a whole synthesis round trip and sound clipped.
                    min_words = 2 if emitted_chunks == 0 else 6
                    min_terminal = 2 if emitted_chunks == 0 else 4
                    should_emit = (
                        (not is_pause and words >= min_words)
                        or (is_terminal and words >= min_terminal)
                        or (is_pause and words >= 10)
                        or (emitted_chunks == 0 and is_pause and words >= 3)
                    )

                    if not should_emit:
                        pending_tts_fragment = (
                            f"{pending_tts_fragment} {candidate}".strip()
                            if pending_tts_fragment
                            else candidate
                        )
                        continue

                    out = (
                        f"{pending_tts_fragment} {candidate}".strip()
                        if pending_tts_fragment
                        else candidate
                    )
                    pending_tts_fragment = ""

                    if _has_spoken_content(out):
                        emitted_chunks += 1
                        await tts_queue.put(out)

        if not stream_interrupted:
            tail = (
                f"{pending_tts_fragment} {buffer.strip()}".strip()
                if pending_tts_fragment
                else buffer.strip()
            )
            if tail and _has_spoken_content(tail):
                emitted_chunks += 1
                await tts_queue.put(tail)

        if stream_interrupted:
            await tts_queue.put(None)
            return ""

        response = full_response.strip()
        if not response:
            print(f"[Session {session.session_id}] [LLM] Empty response.")
            await _handle_generation_failure(session, tts_queue, primary_intent, spoke=False)
            return ""
        session.llm_failures = 0

        # Follow-up questions are owned by the prompt's stage goal (one question
        # per turn); nothing is appended after generation.
        if _response_has_closing(response):
            session.close_after_speaking = True

        # Sentinel to indicate LLM generation is complete
        await tts_queue.put(None)

        session.chat_history.append({"role": "user",      "content": user_input})
        session.chat_history.append({"role": "assistant", "content": response})
        if len(session.chat_history) > 16:
            session.chat_history[:] = session.chat_history[-16:]

        return response

    except asyncio.CancelledError:
        # Interruption path can cancel the pipeline task mid-stream.
        # Ensure TTS worker receives a sentinel and then propagate cancellation.
        try:
            await tts_queue.put(None)
        except Exception:
            pass
        print(f"[Session {session.session_id}] [LLM] Cancelled.")
        raise
    except Exception as e:
        print(f"[Session {session.session_id}] [LLM] Exception: {e}")
        await _handle_generation_failure(session, tts_queue, primary_intent, spoke=emitted_chunks > 0)
        return ""
