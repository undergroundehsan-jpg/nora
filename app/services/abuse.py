"""Abuse detection for inbound speech, in three tiers.

  severe   — threats, wishes of harm, slurs. The call ends immediately.
  directed — insults or profanity aimed at NORA ("fuck you", "stupid bot").
             One calm offer, then the call ends.
  venting  — profanity about the situation, not the agent ("shit, I forgot").
             NOT abuse: the conversation carries on normally.

The split matters in both directions. Missing directed abuse leaves the agent
answering an insult as if it were a question; treating venting as abuse ends a
call with a cooperative customer who just swore at their own forgetfulness.

Extra terms can be added without touching code: drop a JSON file at
data/abuse_terms.json shaped {"severe": [...], "directed": [...]} where each
entry is a regular expression.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
_EXTRA_TERMS_PATH = _BASE_DIR / "data" / "abuse_terms.json"

# Letters people use to mask profanity: f*ck, sh!t, b1tch, a$$hole.
_M = {"a": "[a@4]", "i": "[i1!*]", "o": "[o0*]", "u": "[u*]", "s": "[s$5]", "e": "[e3]"}


def _mask(word: str) -> str:
    """Build a pattern that also matches lightly censored spellings."""
    return "".join(_M.get(ch, ch) for ch in word)


# ── Tier 1: severe — end the call at once ───────────────────────────────────
_SEVERE_PATTERNS = [
    # Harm aimed at the agent or anyone else.
    r"\b(kill|hurt|harm|stab|shoot|beat) your ?self\b",
    r"\byou should (die|kill|rot|burn)\b",
    r"\b(go|just) (die|kill yourself)\b",
    r"\bdrop dead\b",
    r"\bi hope you (die|rot|suffer|burn|get hurt)\b",
    r"\bi('ll| will| am going to| wanna| want to) (kill|hurt|harm|find|beat|shoot|stab|destroy) (you|u)\b",
    r"\bi('ll| will) (come|show up) (to|at) your (office|house|home)\b",
    r"\bwatch your back\b",
    r"\byou('re| are) (dead|finished)\b",
    # Sexual aggression and the most common slurs (deliberately short list;
    # extend through data/abuse_terms.json rather than editing this file).
    r"\brape\b",
    r"\bn[i1]gg(a|er)\b",
    r"\bf[a@]gg?ot\b",
]

# ── Tier 2: directed — insult or profanity aimed at NORA ────────────────────
_INSULT_NOUNS = (
    r"idiot|moron|fool|stupid|dumb|dumbass|jackass|asshole|arsehole|bastard|bitch|"
    r"prick|dickhead|dick|wanker|twat|cunt|scum|loser|liar|thief|fraud|clown|joke|"
    r"useless|worthless|pathetic|garbage|trash|rubbish|nonsense|annoying|disgusting"
)

_DIRECTED_PATTERNS = [
    # Classic directed profanity.
    rf"\b{_mask('fuck')}\w*\s*(you|u|off|yourself)\b",
    r"\bf+\s*u+\s*c+\s*k+\s*(you|u|off)\b",      # "f u c k you"
    r"\bf\W{1,3}k\s*(you|u|off)\b",               # "f**k you"
    r"\b[a-z]\*{2,}\s*(you|u|off)\b",             # "f*** you", "s*** off"
    r"\bstf?u\b", r"\bwtf\b", r"\bgtfo\b",
    rf"\b{_mask('screw')} (you|u|this)\b",
    r"\b(piss|bugger|sod|naff) off\b",
    r"\bget (lost|stuffed)\b",
    r"\bshove it\b",
    r"\bshut (up|the (hell|f\w*) up|it)\b",
    r"\bgo to hell\b",
    r"\bdamn you\b",
    # Insults aimed at the agent or the people behind it.
    rf"\byou('re| are| r)? +(a |an |such a |nothing but a )?({_INSULT_NOUNS})s?\b",
    rf"\byou +({_INSULT_NOUNS})s?\b",
    rf"\b({_INSULT_NOUNS}) +(bot|robot|machine|ai|system|service|company|people|agent|woman|man|lady)\b",
    rf"\byou people are +(a |an )?({_INSULT_NOUNS})s?\b",
    r"\byou (suck|stink)\b",
    r"\b(stupid|useless|bloody|damn|fucking) (bot|robot|machine|ai|recording|system)\b",
    # Romanised Urdu/Hindi abuse, common on Pakistani calls.
    r"\b(bhen?chod|bhenchod|behenchod|madar ?chod|ma ?chod|chutiy?a|chutya|gandu|gaandu|"
    r"harami|haram ?khor|kutta|kuttay|kameena|kaminay|ullu|badtameez|besharam|zaleel|"
    r"bakwas|lanat|teri maa|teri ma|maa ki|gaand|lavde|lode|bhosdi\w*)\b",
]

# ── Strong profanity, which counts as abuse unless the caller is venting ────
_STRONG_PROFANITY = [
    _mask("fuck"), _mask("shit"), _mask("bitch"), "bastard", _mask("asshole"), "arsehole",
    "cunt", "prick", "dickhead", "wanker", "twat", "bollocks", "motherf\\w*", "bullshit",
    "son of a bitch", "piss(ed)? off",
    r"f\*{1,4}k", r"f\*{2,}", r"s\*{2,}t", r"b\*{2,}h",   # censored spellings
]
_STRONG_PROFANITY_RE = re.compile(r"\b(" + "|".join(_STRONG_PROFANITY) + r")\b")
_PROFANITY_STRIP_RE = re.compile(
    r"\b(" + "|".join(_STRONG_PROFANITY) + r")\w*\b"
    r"|\b(damn|bloody|freaking|frigging|goddamn|flipping)\w*\b"
)

# Venting: the profanity is about their own situation, not about NORA.
_VENTING_CONTEXT_RE = re.compile(
    r"\b(i|i'm|im|i've|we|we're|my|our|me)\b.*\b(forgot|thought|paid|pay|late|busy|broke|"
    r"tired|sick|sorry|bill|money|job|work)\b"
    r"|\b(forgot|thought|paid|didn't know|did not know)\b"
    r"|^(oh|ah|aw|ugh|god|jesus|christ)\b"
)

# Mild exclamations that are never abuse on their own.
_MILD_ONLY_RE = re.compile(r"^\W*(damn|dang|darn|crap|heck|bloody hell|god ?damn ?it|jeez|ugh)\W*$")


def _load_extra_terms() -> dict:
    try:
        data = json.loads(_EXTRA_TERMS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _compile(patterns: list[str], extra_key: str) -> re.Pattern:
    extra = _load_extra_terms().get(extra_key) or []
    return re.compile("|".join(patterns + [p for p in extra if isinstance(p, str)]), re.IGNORECASE)


_SEVERE_RE = _compile(_SEVERE_PATTERNS, "severe")
_DIRECTED_RE = _compile(_DIRECTED_PATTERNS, "directed")


def strip_profanity(text: str) -> str:
    """Remove swear words so the rest of the sentence can still be understood.

    "I already fucking paid it" is a payment confirmation with a swear word in
    the middle, not an unclear reply.
    """
    cleaned = _PROFANITY_STRIP_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def is_severe(text: str) -> bool:
    """Threats, wishes of harm or slurs — the call should end immediately."""
    return bool(_SEVERE_RE.search(text or ""))


def is_abusive(text: str) -> bool:
    """True when the words are aimed at NORA rather than at the situation."""
    if not text:
        return False
    if is_severe(text) or _DIRECTED_RE.search(text):
        return True
    if _MILD_ONLY_RE.match(text.strip()):
        return False
    if _STRONG_PROFANITY_RE.search(text):
        # Strong language, but if they are venting about themselves it is not
        # abuse: "shit, I forgot to pay that" is a cooperative customer.
        return not _VENTING_CONTEXT_RE.search(text)
    return False
