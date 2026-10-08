"""Call-flow policy engine for payment/service reminder calls.

This is the single place that decides *what happens next* on a turn. It runs
after intent classification and before the LLM:

  P1 safety      → fixed line, log escalation, end call
  P0 compliance  → fixed line, record suppression, end call
  P2 escalation  → offer a callback from the team (no live transfer yet)
  P3 right party → no disclosure, get a callback time
  P4 soft no     → one callback offer, then close
  control        → hold / repeat / slow down / goodbye / AI disclosure
  P5 normal flow → hand over to the LLM (kind == "LLM")

Everything here is deterministic and synchronous so it can be unit tested
without audio, network, or an event loop. Side effects that must be written
to disk are returned as `records` for the caller to persist.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# Action kinds
LLM = "LLM"        # continue with the normal LLM pipeline
SAY = "SAY"        # speak a fixed line and keep the call open
REPEAT = "REPEAT"  # re-speak the last assistant reply (text already filled in)
END = "END"        # speak a fixed line, then end the call

# Hold-mode timings (seconds of silence while the caller asked us to wait)
HOLD_CHECKIN_SECS = 30.0
HOLD_MAX_SECS = 90.0

# Maximum times we ask for a callback time before closing with a generic promise
MAX_CALLBACK_ASKS = 2

_TEMPLATES = {
    # ── P1 safety ──
    "DISTRESS": (
        "I'm really sorry you're dealing with that. If you're in danger or need urgent help, "
        "please call 9 1 1 right away, or call or text 9 8 8 to reach the crisis line. "
        "I'll end this call now so you can get help. Take care."
    ),
    # ── P0 compliance ──
    "DO_NOT_CALL": "Understood. I'll remove this number from our list. Goodbye.",
    "WRONG_NUMBER": (
        "Sorry about that, I must have the wrong number. I'll update our records. Have a good day."
    ),
    "ATTORNEY": (
        "Understood, thank you for letting me know. I'll note that you're represented and we'll stop these reminder calls. Goodbye."
    ),
    "BANKRUPTCY": (
        "Thank you for letting me know. I'll note that on the account and stop these reminder calls. Goodbye."
    ),
    "DECEASED": (
        "I'm very sorry for your loss. I'll update our records so you don't receive more of these calls. Take care."
    ),
    "LANGUAGE_BARRIER": (
        "Sorry for the trouble. We'll follow up another way. Goodbye."
    ),
    # ── Abuse ──
    "ABUSE_FIRST": "I'll let you go. Would you like a callback another time?",
    "ABUSE_END": "I'll end the call here. Have a good day.",
    # Threats and harm aimed at the agent: close immediately, offer nothing.
    "ABUSE_SEVERE_END": "I'm going to end the call here. Goodbye.",
    # ── P2 escalation (callback, no live transfer) ──
    "TRANSFER_TO_HUMAN": (
        "Of course. I can't transfer you, but someone from our team can call you back. "
        "What time works?"
    ),
    "DISPUTE": (
        "Thanks for telling me, I'll stop the reminder there. "
        "Someone from our team can call you to review it. What time works?"
    ),
    "HARDSHIP": (
        "I understand, and there's no pressure at all. "
        "Someone can call you to talk through options. What time works?"
    ),
    "PAYMENT_PLAN": (
        "Our team can set that up for you. When's a good time for them to call?"
    ),
    "PAY_NOW": (
        "For your security I can't take payment on this call. "
        "You can pay using the number on your bill, or I can have someone call you. "
        "Which would you prefer?"
    ),
    "BAD_EXPERIENCE": (
        "I'm sorry to hear that. Someone from our team will follow up with you. "
        "What time works?"
    ),
    # ── P3 right party ──
    "THIRD_PARTY": (
        "No problem. When would be a good time to reach them?"
    ),
    # ── P4 soft objections ──
    "TOO_BUSY": "No problem. When would be a better time for a quick call back?",
    "NOT_INTERESTED": "Understood. Would you like a call back another time, or should I stop these reminders?",
    "SOFT_CLOSE": "Understood. Thanks for your time, and have a good day.",
    "CANNOT_PROCEED": "Understood. Thanks for your time, and have a good day.",
    # ── Outcomes ──
    "ALREADY_PAID": (
        "Thanks for letting me know, I'll note it as paid. Have a great day!"
    ),
    "ASK_TIME": "Sure. What day and time works best for you?",
    "ASK_TIME_THIRD_PARTY": "When would be a good day and time to reach them?",
    "CALLBACK_CONFIRMED": "Thanks. I've noted {time}. We'll reach out then. Have a great day!",
    "CALLBACK_NO_TIME": (
        "No problem. Someone from our team will reach out during business hours. Have a great day!"
    ),
    "CALLBACK_DECLINED": "No problem at all. Thanks for your time, and have a good day.",
    "PARTIAL_OPT_OUT": (
        "Got it, I've noted that preference and we'll follow it going forward. Thanks, and have a good day."
    ),
    # ── Conversation control ──
    "HOLD": "Sure, take your time.",
    "HOLD_CHECKIN": "I'm still here whenever you're ready.",
    "HOLD_TIMEOUT": "It sounds like now isn't a good time. We'll try you another time. Goodbye.",
    "SLOW_DOWN_PREFIX": "Sure, I'll slow down.",
    "REPEAT_FALLBACK": "Sorry about that. I'm calling with a quick reminder about your account.",
    "REPEAT_LIMIT": (
        "Sorry, the line isn't great. When's a good time for someone to call you back?"
    ),
    "LOOP_BREAK": (
        "I don't want to keep asking the same thing. "
        "When's a good time for someone to call you back?"
    ),
    "AI_DISCLOSURE_FIRST": (
        "Yes, I'm NORA, an automated assistant calling for {org}. "
        "If you'd rather talk to a person, I can have someone call you back. Otherwise, I'll keep this quick."
    ),
    "AI_DISCLOSURE_REPEAT": (
        "Yes, this is an automated call. Would you like someone from our team to call you back instead?"
    ),
    "GOODBYE": "Thanks for your time. Have a great day!",
    "TECH_TROUBLE": (
        "Sorry, we're having some technical trouble on our end. We'll try you again another time. Goodbye."
    ),
    "LLM_RETRY": "Sorry, I missed that. Could you say it one more time?",
    # ── Telephony ──
    "VOICEMAIL": (
        "Hi, this is NORA calling from {org}. Please call us back at the number on your account. Thank you."
    ),
}

# Dispositions written to the session log
DISPOSITIONS = {
    "DISTRESS": "SAFETY_ESCALATION",
    "DO_NOT_CALL": "DO_NOT_CALL",
    "WRONG_NUMBER": "WRONG_NUMBER",
    "ATTORNEY": "ATTORNEY_REPRESENTED",
    "BANKRUPTCY": "BANKRUPTCY",
    "DECEASED": "DECEASED",
    "LANGUAGE_BARRIER": "LANGUAGE_BARRIER",
}

_P0_INTENTS = {"DO_NOT_CALL", "WRONG_NUMBER", "ATTORNEY", "BANKRUPTCY", "DECEASED", "LANGUAGE_BARRIER"}


def is_severe_abuse(text: str) -> bool:
    """Threats, wishes of harm or slurs end the call without an offer."""
    from app.services.abuse import is_severe
    return is_severe(_norm(text))
_ESCALATION_INTENTS = {"TRANSFER_TO_HUMAN", "DISPUTE", "HARDSHIP", "PAYMENT_PLAN", "BAD_EXPERIENCE"}
_THIRD_PARTY_INTENTS = {"THIRD_PARTY", "IDENTITY_DENIAL", "NOT_DECISION_MAKER"}


@dataclass
class PolicyAction:
    kind: str
    text: str = ""
    disposition: Optional[str] = None
    reason: str = ""
    records: list = field(default_factory=list)
    # Template name when the line is fixed text, so it can play from
    # audio_cache/tpl_<name>.pcm instead of a live TTS call.
    cache_key: Optional[str] = None


def is_cacheable(name: str) -> bool:
    """True when a template has no placeholders, so it can be pre-recorded."""
    return "{" not in _TEMPLATES.get(name, "{")


def cacheable_templates() -> dict:
    """{template name: text} for every line that can be pre-recorded."""
    return {name: text for name, text in _TEMPLATES.items() if is_cacheable(name)}


def template(name: str, **kwargs) -> str:
    return _TEMPLATES[name].format(**kwargs)


def _org(session) -> str:
    lead = getattr(session, "lead_context", None) or {}
    return lead.get("practice_name") or "our office"


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


# ── Answer shape helpers ──────────────────────────────────────────────────────

_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|alright|please|correct|absolutely|definitely|"
    r"go ahead|sounds good|that works|that'?s fine|that would be great|speaking|this is (she|he|him|her|me)|"
    r"that'?s me|it is|i am)\b"
)
_NEGATIVE_RE = re.compile(
    r"\b(no(?!\s*(?:problem|worries|doubt))|nope|nah|not really|never ?mind|don't bother|no thanks|no thank you|forget it)\b"
)


def is_affirmative(text: str) -> bool:
    t = _norm(text)
    return bool(_AFFIRMATIVE_RE.search(t)) and not bool(_NEGATIVE_RE.search(t))


def is_negative(text: str) -> bool:
    return bool(_NEGATIVE_RE.search(_norm(text)))


_WEEKDAYS = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
_NUMBER_WORDS = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_TIME_PATTERNS = [
    rf"\b(next|this|on|every)?\s*{_WEEKDAYS}(\s+(morning|afternoon|evening|night))?\b",
    r"\b(tomorrow|today|tonight|this (morning|afternoon|evening|week)|next week|later today|end of (the )?(day|week))(\s+(morning|afternoon|evening|night))?\b",
    r"\b(?<!good )(in the )?(morning|afternoon|evening)\b",
    # Digits after a preposition: "at 5", "after 3:30", "by 11 am".
    r"\b(at|after|before|around|by)\s+\d{1,2}(:\d{2})?(\s*(am|pm|a\.m\.|p\.m\.|o'clock))?\b",
    # Number words only with a clear time marker, or at the end of a phrase:
    # "around five pm", "after three." — but not "at one point" / "by two people".
    rf"\b(at|after|before|around|by)\s+({_NUMBER_WORDS}|noon|midday)(\s*(am|pm|a\.m\.|p\.m\.|o'clock)|\s*(?=[.,!?]|$))",
    r"\b\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.|o'clock)\b",
    r"\b\d{1,2}:\d{2}\b",
    r"\bin (an|a|one|two|three|four|five|a few|\d+) (minutes?|hours?|days?|weeks?)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?\b",
    r"\b\d{1,2}/\d{1,2}\b",
]
_ANYTIME_RE = re.compile(r"\b(any ?time|whenever( works| is fine)?|doesn't matter when)\b")
# A time right after a negation is a time to avoid: "not tomorrow", "no, not friday".
_NEGATED_BEFORE_RE = re.compile(r"(\bnot|\bno|n't|\bexcept|\bbut not)\s+(on\s+|at\s+|this\s+|next\s+)?$")

ANY_TIME = "any time"


def extract_callback_time(text: str) -> Optional[str]:
    """Return the time-like phrases in `text` joined in order, or None.

    Times directly after a negation are ignored ("not tomorrow, friday" → "friday").
    A caller who says "anytime" / "whenever" gets ANY_TIME.
    """
    t = _norm(text)
    spans: list[tuple[int, int]] = []
    for pattern in _TIME_PATTERNS:
        for m in re.finditer(pattern, t):
            if not m.group(0).strip():
                continue
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            if _NEGATED_BEFORE_RE.search(t[max(0, start - 14):start]):
                continue
            spans.append((start, m.end()))
    if not spans:
        return ANY_TIME if _ANYTIME_RE.search(t) else None
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    phrase = " ".join(t[s_:e].strip(" .,!?") for s_, e in merged)
    return re.sub(r"^(on|at|around|by)\s+", "", phrase).strip() or None


_UNKNOWN_RE = re.compile(r"\b(i )?(don't|do not|dont) know\b|\bnot sure\b|\bno idea\b|\bcan't say\b")


def is_unknown(text: str) -> bool:
    return bool(_UNKNOWN_RE.search(_norm(text)))


_IDENTITY_QUESTION_RE = re.compile(
    r"(is this [a-z .']+\?"
    r"|am i speaking (with|to)"
    r"|speaking with [a-z .']+\?"
    r"|may i (please )?speak (with|to)"
    r"|could i speak (with|to)"
    r"|can i speak (with|to)"
    r"|are you [a-z .']+\?"
    r"|who am i speaking"
    r"|connect me (with|to)"
    r"|put (him|her|them) on"
    r"|is [a-z .']+ (available|there|around)"
    r"|right person"
    r"|is that you)",
    re.IGNORECASE,
)
# Phrases that confirm identity on their own, whatever NORA happened to ask.
_STRONG_IDENTITY_RE = re.compile(
    r"\b(speaking|that'?s me|it'?s me|this is (she|he|him|her|me)|"
    r"you'?(re| are) speaking (with|to) (her|him|me)|you'?ve got (her|him|me))\b"
)
_IDENTITY_CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|speaking|this is (she|he|him|her|me)|that'?s me|it'?s me|it is|"
    r"i am|you got (her|him|me)|you are|you're speaking with)\b"
)


def _last_assistant(session) -> str:
    for msg in reversed(getattr(session, "chat_history", []) or []):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def _recent_assistant(session, n: int = 2) -> list[str]:
    out = []
    for msg in reversed(getattr(session, "chat_history", []) or []):
        if msg.get("role") == "assistant":
            out.append(msg.get("content") or "")
            if len(out) == n:
                break
    return out


def _similar(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z']+", a.lower()))
    wb = set(re.findall(r"[a-z']+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_repeating_itself(session, threshold: float = 0.8) -> bool:
    """True when the last two assistant turns asked essentially the same question."""
    recent = _recent_assistant(session, 2)
    if len(recent) < 2 or "?" not in recent[0] or "?" not in recent[1]:
        return False
    return _similar(recent[0], recent[1]) >= threshold


# ── Telephony prompts (voicemail / IVR) ───────────────────────────────────────

_VOICEMAIL_RE = re.compile(
    r"(leave (a |your )?message|after the (tone|beep)|voicemail|mailbox|not available to take your call|"
    r"record your message)"
)
_IVR_RE = re.compile(r"\b(press|dial) (one|two|three|four|five|six|seven|eight|nine|zero|\d)\b|\bmain menu\b")


def classify_telephony_prompt(transcript: str) -> Optional[str]:
    """Return "VOICEMAIL", "IVR", or None for carrier/system audio heard at call start."""
    t = _norm(transcript)
    if not t:
        return None
    if _VOICEMAIL_RE.search(t):
        return "VOICEMAIL"
    if _IVR_RE.search(t):
        return "IVR"
    return None


# ── State updates ─────────────────────────────────────────────────────────────

def update_identity(session, transcript: str, intent: str) -> None:
    """Mark identity verified when the caller confirms after an identity question."""
    if session.identity_verified:
        return
    if intent in _THIRD_PARTY_INTENTS | _P0_INTENTS:
        return
    t = _norm(transcript)
    if is_negative(t):
        return

    # "Speaking", "that's me" and the like are unambiguous on their own.
    if _STRONG_IDENTITY_RE.search(t):
        session.identity_verified = True
        return

    # A bare "yes" only confirms identity when NORA actually asked who it is
    # (never after "is now a good time?").
    question = _IDENTITY_QUESTION_RE.search(_last_assistant(session))
    if not question or "time" in question.group(0).lower():
        return
    if _IDENTITY_CONFIRM_RE.search(t):
        session.identity_verified = True


_REMINDER_CONTENT_RE = re.compile(
    r"\b(remind|reminder|due|payment|balance|bill|appointment|account|service|paid)\b", re.IGNORECASE
)


def note_assistant_turn(session, stage_at_start: str, response: str = "") -> None:
    """Advance slots after an LLM reply finished for the given stage."""
    if (
        stage_at_start == "REMIND"
        and not session.reminder_delivered
        and (not response or _REMINDER_CONTENT_RE.search(response))
    ):
        session.reminder_delivered = True
        session.pitch_delivered = True


def hold_status(session, now: Optional[float] = None) -> Optional[str]:
    """While on hold: return "CHECKIN" or "TIMEOUT" when due, else None."""
    if not getattr(session, "hold_active", False):
        return None
    elapsed = (now or time.time()) - session.hold_started_at
    if elapsed >= HOLD_MAX_SECS:
        return "TIMEOUT"
    if elapsed >= HOLD_CHECKIN_SECS and not session.hold_checkin_sent:
        return "CHECKIN"
    return None


# ── Decision ──────────────────────────────────────────────────────────────────

def _end(name: str, disposition: str, reason: str, records=None, **fmt) -> PolicyAction:
    return PolicyAction(END, template(name, **fmt), disposition, reason, records or [],
                        cache_key=name if is_cacheable(name) else None)


def _say(name: str, reason: str, **fmt) -> PolicyAction:
    return PolicyAction(SAY, template(name, **fmt), reason=reason,
                        cache_key=name if is_cacheable(name) else None)


def _callback_record(session, reason: str, when: Optional[str]) -> dict:
    return {"type": "callback_request", "reason": reason, "requested_time": when}


def _start_callback(session, reason: str, template_name: str, transcript: str) -> PolicyAction:
    """Offer a callback. If the caller already named a time, confirm and close at once."""
    when = extract_callback_time(transcript)
    if reason in _ESCALATION_INTENTS or reason == "PAY_NOW":
        session.escalation_reason = session.escalation_reason or reason
    if when:
        return _confirm_callback(session, reason, when)
    session.pending_callback_reason = reason
    session.callback_asks = 0
    return PolicyAction(SAY, template(template_name, org=_org(session)), reason=reason,
                        cache_key=template_name if is_cacheable(template_name) else None)


# Replies that mean "I'm answering your callback question" (a time may be inside).
_CALLBACK_ANSWER_INTENTS = {"UNCLEAR", "INTERESTED", "CALL_BACK_LATER", "RESCHEDULE", "TOO_BUSY", "GOODBYE"}
# Conversation-control intents keep the callback question open and are handled normally.
_KEEP_PENDING_INTENTS = {"HOLD_REQUEST", "REPEAT_REQUEST", "SLOW_DOWN"}
# Declining one of these offers means "keep going", not "end the call".
_CONTINUE_ON_DECLINE = {"AI_DISCLOSURE"}
_END_REQUEST_RE = re.compile(r"\b(end (it|the call|this call)|hang up|i'?m done|that'?s all|we'?re done)\b")
_SOFT_CLOSE_REASONS = {"TOO_BUSY", "NOT_INTERESTED", "THIRD_PARTY", "ABUSE", "LOOP", "REPEAT_LIMIT"}


def _confirm_callback(session, reason: str, when: str) -> PolicyAction:
    session.pending_callback_reason = None
    session.outcome = "CALLBACK"
    record = [_callback_record(session, reason, None if when == ANY_TIME else when)]
    if when == ANY_TIME:
        return _end("CALLBACK_NO_TIME", "CALLBACK_NO_TIME", reason, record)
    session.callback_time = when
    return _end("CALLBACK_CONFIRMED", "CALLBACK_SCHEDULED", reason, record, time=when)


def _resolve_pending_callback(session, transcript: str, intent: str) -> Optional[PolicyAction]:
    """Handle a reply while NORA is waiting for a callback day/time.

    Returns an action, or None to let the intent be handled normally (the
    pending question is cleared unless it's a hold/repeat/slow-down).
    """
    reason = session.pending_callback_reason
    t = _norm(transcript)

    if intent in _KEEP_PENDING_INTENTS:
        return None

    # They restated the problem (dispute, wanting a person) while we are already
    # waiting for a time. Re-offering the callback would repeat the same line, so
    # just ask for the time again, briefly.
    if intent in _ESCALATION_INTENTS or intent == "PAY_NOW":
        session.escalation_reason = session.escalation_reason or intent
        return _say("ASK_TIME", reason)

    # "...or stop these reminder calls?" → "just stop them" is a full opt-out.
    if reason == "NOT_INTERESTED" and re.search(r"\b(stop|remove|no more|opt out)\b", t):
        session.pending_callback_reason = None
        return _end("DO_NOT_CALL", DISPOSITIONS["DO_NOT_CALL"], "DO_NOT_CALL",
                    [{"type": "suppression", "reason": "DO_NOT_CALL"}])

    if intent not in _CALLBACK_ANSWER_INTENTS:
        # A new topic (already paid, dispute, a question...): drop the ask.
        session.pending_callback_reason = None
        return None

    when = extract_callback_time(transcript)
    if when:
        return _confirm_callback(session, reason, when)

    if intent == "TOO_BUSY":
        # Still busy and no time given: let the soft-objection rule close politely.
        session.pending_callback_reason = None
        return None

    if _END_REQUEST_RE.search(t):
        session.pending_callback_reason = None
        return _end("SOFT_CLOSE", "DECLINED", reason)

    if is_unknown(transcript):
        session.pending_callback_reason = None
        if reason == "THIRD_PARTY":
            return _end("CALLBACK_DECLINED", "CALLBACK_NO_TIME", reason)
        return _confirm_callback(session, reason, ANY_TIME)

    if intent in {"GOODBYE", "NOT_INTERESTED"} or is_negative(transcript):
        session.pending_callback_reason = None
        if reason in _CONTINUE_ON_DECLINE:
            # "No thanks, just keep going" → the LLM picks up the next question.
            return PolicyAction(LLM, reason=reason)
        if reason in _SOFT_CLOSE_REASONS:
            return _end("SOFT_CLOSE", "DECLINED", reason)
        return _end("CALLBACK_DECLINED", "ESCALATION_DECLINED", reason)

    session.callback_asks += 1
    if session.callback_asks > MAX_CALLBACK_ASKS:
        return _confirm_callback(session, reason, ANY_TIME)
    ask = "ASK_TIME_THIRD_PARTY" if reason == "THIRD_PARTY" else "ASK_TIME"
    return _say(ask, reason)


def decide(session, transcript: str, intent: str) -> PolicyAction:
    """Pick the action for this turn. Mutates call-state slots on `session`."""
    # Any real reply ends hold mode.
    if session.hold_active:
        session.hold_active = False

    update_identity(session, transcript, intent)

    # ── P1 safety ──
    if intent == "DISTRESS":
        return _end("DISTRESS", DISPOSITIONS["DISTRESS"], intent,
                    [{"type": "escalation", "reason": "DISTRESS"}])

    # ── P0 compliance ──
    if intent in _P0_INTENTS:
        records = []
        if intent in {"DO_NOT_CALL", "ATTORNEY", "BANKRUPTCY", "DECEASED", "WRONG_NUMBER"}:
            records.append({"type": "suppression", "reason": intent})
        return _end(intent, DISPOSITIONS[intent], intent, records)

    # ── Abuse ──
    if intent == "ABUSE":
        session.abuse_count += 1
        if is_severe_abuse(transcript):
            return _end("ABUSE_SEVERE_END", "ENDED_ABUSE", intent)
        if session.abuse_count >= 2:
            return _end("ABUSE_END", "ENDED_ABUSE", intent)
        session.pending_callback_reason = "ABUSE"
        session.callback_asks = 0
        return _say("ABUSE_FIRST", intent)

    # ── Waiting for a callback time ──
    if session.pending_callback_reason:
        action = _resolve_pending_callback(session, transcript, intent)
        if action is not None:
            return action

    # ── P2 escalation → callback ──
    if intent in _ESCALATION_INTENTS:
        return _start_callback(session, intent, intent, transcript)
    if intent == "PAY_NOW":
        session.escalation_reason = session.escalation_reason or intent
        session.pending_callback_reason = intent
        session.callback_asks = 0
        return _say("PAY_NOW", intent)

    # ── P3 right party ──
    if intent in _THIRD_PARTY_INTENTS and not session.identity_verified:
        return _start_callback(session, "THIRD_PARTY", "THIRD_PARTY", transcript)

    # ── Outcomes ──
    if intent == "ALREADY_PAID":
        session.outcome = "PAID"
        return _end("ALREADY_PAID", "ALREADY_PAID", intent)
    if intent == "PARTIAL_OPT_OUT":
        session.preferences["contact_preference"] = transcript.strip()
        session.outcome = "PREFERENCE_UPDATED"
        return _end("PARTIAL_OPT_OUT", "PREFERENCE_UPDATED", intent,
                    [{"type": "preference", "value": transcript.strip()}])
    if intent in {"RESCHEDULE", "CALL_BACK_LATER"}:
        return _start_callback(session, intent, "ASK_TIME", transcript)

    # ── P4 soft objections: one offer, then close ──
    if intent in {"TOO_BUSY", "NOT_INTERESTED"}:
        session.soft_objection_count += 1
        if session.abuse_count >= 1 or session.soft_objection_count >= 2:
            return _end("SOFT_CLOSE", "DECLINED", intent)
        return _start_callback(session, intent, intent, transcript)
    if intent == "CANNOT_PROCEED":
        return _end("CANNOT_PROCEED", "DECLINED", intent)

    # ── Conversation control ──
    if intent == "HOLD_REQUEST":
        session.hold_active = True
        session.hold_started_at = time.time()
        session.hold_checkin_sent = False
        return _say("HOLD", intent)

    if intent == "REPEAT_REQUEST":
        session.repeat_request_count += 1
        if session.repeat_request_count >= 3:
            session.pending_callback_reason = "REPEAT_LIMIT"
            session.callback_asks = 0
            return _say("REPEAT_LIMIT", intent)
        last = session.last_response or _last_assistant(session)
        return PolicyAction(REPEAT, last or template("REPEAT_FALLBACK"), reason=intent)

    if intent == "SLOW_DOWN":
        current = session.tts_time_scale or 1.0
        session.tts_time_scale = min(1.6, round(current * 1.2, 2))
        last = session.last_response or _last_assistant(session)
        text = template("SLOW_DOWN_PREFIX") + (f" {last}" if last else "")
        return PolicyAction(REPEAT, text, reason=intent)

    if intent == "ASK_IF_AI":
        if not session.ai_disclosed:
            session.ai_disclosed = True
            return PolicyAction(SAY, template("AI_DISCLOSURE_FIRST", org=_org(session)), reason=intent)
        session.pending_callback_reason = "AI_DISCLOSURE"
        session.callback_asks = 0
        return _say("AI_DISCLOSURE_REPEAT", intent)

    if intent == "GOODBYE":
        return _end("GOODBYE", session.outcome or "CALLER_ENDED", intent)

    # ── Anti-loop: never ask the same question a third time ──
    if is_repeating_itself(session):
        session.pending_callback_reason = "LOOP"
        session.callback_asks = 0
        return _say("LOOP_BREAK", "LOOP")

    return PolicyAction(LLM, reason=intent)
