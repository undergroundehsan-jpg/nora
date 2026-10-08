"""Lightweight intent classifier for outbound payment/service reminder calls.

Order of precedence:
  1. Deterministic regex heuristics (compliance hard-stops are checked first,
     so nothing else can shadow a do-not-call / attorney / distress phrase).
  2. Numeric/date answer shortcut (treated as an answer, never a refusal).
  3. A small, fast LLM classifier for everything else, with guardrails.
"""
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from groq import AsyncGroq

from app.services.abuse import is_abusive, strip_profanity

load_dotenv()
_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Fast model for classification (cheapest, lowest latency) ───
# Override with GROQ_CLASSIFIER_MODEL when Groq retires a model.
# Note: openai/gpt-oss-* models spend the token budget on hidden reasoning and
# return an empty message at these limits — don't use them here.
CLASSIFIER_MODEL = os.getenv("GROQ_CLASSIFIER_MODEL", "qwen/qwen3.8-27b").strip()
_CLASSIFIER_MODEL = CLASSIFIER_MODEL  # backwards-compatible alias


async def warm_up_classifier_connection() -> None:
    """Open this client's HTTPS connection early (no tokens generated)."""
    try:
        await _client.models.list()
    except Exception:
        pass


class ContentTier(str, Enum):
    COMMAND = "COMMAND"
    DEFINITION = "DEFINITION"
    EXPLANATION = "EXPLANATION"
    OBJECTION = "OBJECTION"


class OutreachIntent(str, Enum):
    # ── Core conversation intents ──
    GATEKEEPER = "GATEKEEPER"
    INTERESTED = "INTERESTED"            # cooperative / open ("okay", "sure", "go ahead")
    ASK_MORE_INFO = "ASK_MORE_INFO"
    ASK_SOURCE = "ASK_SOURCE"
    ASK_IF_AI = "ASK_IF_AI"
    EMAIL_REQUEST = "EMAIL_REQUEST"      # wants details in writing
    CALL_BACK_LATER = "CALL_BACK_LATER"

    # ── Reminder outcomes ──
    ALREADY_PAID = "ALREADY_PAID"
    DISPUTE = "DISPUTE"
    HARDSHIP = "HARDSHIP"
    PAYMENT_PLAN = "PAYMENT_PLAN"
    PAY_NOW = "PAY_NOW"
    RESCHEDULE = "RESCHEDULE"

    # ── Conversation control ──
    HOLD_REQUEST = "HOLD_REQUEST"
    REPEAT_REQUEST = "REPEAT_REQUEST"
    SLOW_DOWN = "SLOW_DOWN"
    GOODBYE = "GOODBYE"
    PARTIAL_OPT_OUT = "PARTIAL_OPT_OUT"

    # ── Name / Identity intents ──
    FAKE_NAME = "FAKE_NAME"
    CLARIFY_NAME = "CLARIFY_NAME"
    NOT_DECISION_MAKER = "NOT_DECISION_MAKER"
    IDENTITY_DENIAL = "IDENTITY_DENIAL"
    THIRD_PARTY = "THIRD_PARTY"

    # ── Soft objections ──
    NOT_INTERESTED = "NOT_INTERESTED"
    TOO_BUSY = "TOO_BUSY"
    BAD_EXPERIENCE = "BAD_EXPERIENCE"
    CANNOT_PROCEED = "CANNOT_PROCEED"

    # ── Trust / credibility intents ──
    TRUST_CONCERN = "TRUST_CONCERN"
    PRIVACY_CONCERN = "PRIVACY_CONCERN"

    # ── Compliance / safety hard-stops ──
    DO_NOT_CALL = "DO_NOT_CALL"
    WRONG_NUMBER = "WRONG_NUMBER"
    LANGUAGE_BARRIER = "LANGUAGE_BARRIER"
    ATTORNEY = "ATTORNEY"
    BANKRUPTCY = "BANKRUPTCY"
    DECEASED = "DECEASED"
    DISTRESS = "DISTRESS"
    ABUSE = "ABUSE"

    # ── Escalation / fallback ──
    TRANSFER_TO_HUMAN = "TRANSFER_TO_HUMAN"
    END_CALL = "END_CALL"
    UNCLEAR = "UNCLEAR"

    # ── Legacy (billing-audit era). Never emitted by the classifier anymore;
    # kept so older logs/callers that reference them still resolve. ──
    ASK_AUDIT_DEFINITION = "ASK_AUDIT_DEFINITION"
    ASK_AUDIT_PROCESS = "ASK_AUDIT_PROCESS"
    ASK_COMPLIANCE_BAA = "ASK_COMPLIANCE_BAA"
    ALREADY_HAVE_BILLER = "ALREADY_HAVE_BILLER"
    HAPPY_WITH_PROVIDER = "HAPPY_WITH_PROVIDER"
    ALREADY_AUDITED = "ALREADY_AUDITED"
    NO_BUDGET = "NO_BUDGET"


# Maps each tier to (max_words, max_tokens) for the main LLM.
# max_words is a soft guideline in the prompt; max_tokens is set
# high enough that the model is not forcibly cut off mid-sentence.
_BASE_TIER_LIMITS = {
    ContentTier.COMMAND:     (42, 130),
    ContentTier.DEFINITION:  (56, 170),
    ContentTier.EXPLANATION: (72, 220),
    ContentTier.OBJECTION:   (50, 160),
}

# Legacy alias used by callers that don't pass turn_count.
TIER_LIMITS = _BASE_TIER_LIMITS

# Default if the classifier fails
DEFAULT_TIER = ContentTier.COMMAND


def get_tier_limits(tier: ContentTier, turn_count: int = 0) -> tuple[int, int]:
    """Return (max_words, max_tokens) scaled by conversation depth."""
    words, tokens = _BASE_TIER_LIMITS[tier]
    if turn_count >= 5:
        words = int(words * 1.10)
        tokens = int(tokens * 1.10)
    return words, tokens


@dataclass
class IntentResult:
    """Result of the lightweight intent classification."""
    is_exit: bool
    primary_intent: OutreachIntent
    tier: ContentTier
    counts_as_refusal: bool
    max_words: int
    max_tokens: int
    latency_ms: float


# ── Intents that count toward the refusal limit (and the frontend tracker) ──
REFUSAL_INTENTS = {
    OutreachIntent.NOT_INTERESTED,
    OutreachIntent.TOO_BUSY,
    OutreachIntent.CANNOT_PROCEED,
    OutreachIntent.BAD_EXPERIENCE,
    OutreachIntent.WRONG_NUMBER,
    OutreachIntent.DO_NOT_CALL,
    OutreachIntent.LANGUAGE_BARRIER,
    # Legacy
    OutreachIntent.ALREADY_HAVE_BILLER,
    OutreachIntent.HAPPY_WITH_PROVIDER,
    OutreachIntent.ALREADY_AUDITED,
    OutreachIntent.NO_BUDGET,
}

# ── Intents that end the call (the policy engine speaks a fixed line first) ──
IMMEDIATE_EXIT_INTENTS = {
    OutreachIntent.DO_NOT_CALL,
    OutreachIntent.LANGUAGE_BARRIER,
    OutreachIntent.WRONG_NUMBER,
    OutreachIntent.ATTORNEY,
    OutreachIntent.BANKRUPTCY,
    OutreachIntent.DECEASED,
    OutreachIntent.DISTRESS,
}

HIGH_CONFIDENCE_HEURISTIC_INTENTS = {
    OutreachIntent.DO_NOT_CALL,
    OutreachIntent.PARTIAL_OPT_OUT,
    OutreachIntent.WRONG_NUMBER,
    OutreachIntent.LANGUAGE_BARRIER,
    OutreachIntent.ATTORNEY,
    OutreachIntent.BANKRUPTCY,
    OutreachIntent.DECEASED,
    OutreachIntent.DISTRESS,
    OutreachIntent.ABUSE,
    OutreachIntent.TRANSFER_TO_HUMAN,
    OutreachIntent.HOLD_REQUEST,
    OutreachIntent.REPEAT_REQUEST,
    OutreachIntent.SLOW_DOWN,
    OutreachIntent.GOODBYE,
    OutreachIntent.ALREADY_PAID,
    OutreachIntent.DISPUTE,
    OutreachIntent.HARDSHIP,
    OutreachIntent.PAYMENT_PLAN,
    OutreachIntent.PAY_NOW,
    OutreachIntent.RESCHEDULE,
    OutreachIntent.CALL_BACK_LATER,
    OutreachIntent.EMAIL_REQUEST,
    OutreachIntent.ASK_IF_AI,
    OutreachIntent.TRUST_CONCERN,
    OutreachIntent.PRIVACY_CONCERN,
    OutreachIntent.ASK_SOURCE,
    OutreachIntent.IDENTITY_DENIAL,
    OutreachIntent.THIRD_PARTY,
    OutreachIntent.GATEKEEPER,
    OutreachIntent.NOT_DECISION_MAKER,
    OutreachIntent.TOO_BUSY,
    OutreachIntent.NOT_INTERESTED,
    OutreachIntent.ASK_MORE_INFO,
    OutreachIntent.INTERESTED,
}

WEAKER_MODEL_INTENTS = {
    OutreachIntent.UNCLEAR,
    OutreachIntent.ASK_MORE_INFO,
    OutreachIntent.ASK_SOURCE,
    OutreachIntent.INTERESTED,
    OutreachIntent.FAKE_NAME,
    OutreachIntent.CLARIFY_NAME,
}


def _parse_bool(value, default: bool = False) -> bool:
    """Safely parse booleans from LLM JSON outputs (including string booleans)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return default


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", text))


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _is_social_smalltalk(text: str) -> bool:
    """Detect greeting/check-in replies that should never count as refusal."""
    social_patterns = [
        r"\bhow are you\b",
        r"\bwhat about you\b",
        r"\bhow('s| is) it going\b",
        r"\bit('s| is) going (good|great|fine|okay|ok|well)\b",
        r"\bi('?m| am) (doing )?(good|great|fine|okay|ok|well)\b",
        r"\bdoing (good|great|fine|okay|ok|well)\b",
        r"\bi('?m| am) good\b",
        r"\bthanks?( you)?\b",
        r"\bthank you for\b",
    ]

    if not _any(social_patterns, text):
        return False

    refusal_within_social = [
        r"\bno thanks\b", r"\bno thank you\b", r"\bnah\b",
        r"\bpass\b", r"\bnot interested\b", r"\bdon't need\b",
        r"\bnot looking\b", r"\bdon't want\b", r"\bnot for (us|me)\b",
    ]
    if _any(refusal_within_social, text):
        return False

    objection_keywords = [
        "not interested", "don't call", "do not call", "too busy",
        "paid", "payment", "bill", "owe", "afford",
    ]
    if any(keyword in text for keyword in objection_keywords):
        return False

    return True


_NUMERIC_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "dollar", "dollars", "and",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _looks_like_numeric_answer(text: str) -> bool:
    """Return True when the transcript looks like a numeric/date answer."""
    if not text:
        return False

    percent_clean = re.sub(r"[^0-9%a-z ]", "", text.lower())
    if re.search(r"\b\d{1,3}%\b", percent_clean):
        return True

    if re.fullmatch(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text):
        return True
    if re.fullmatch(r"\$?\d+(?:\.\d+)?", text):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return True

    tokens = re.findall(r"[a-z]+|\d+", text)
    if not tokens:
        return False
    return all(token.isdigit() or token in _NUMERIC_WORDS for token in tokens)


def _intent_to_default_tier(intent: OutreachIntent) -> ContentTier:
    if intent in {OutreachIntent.ASK_SOURCE, OutreachIntent.ASK_MORE_INFO, OutreachIntent.ASK_IF_AI}:
        return ContentTier.DEFINITION
    if intent in {
        OutreachIntent.DISPUTE,
        OutreachIntent.HARDSHIP,
        OutreachIntent.TRUST_CONCERN,
        OutreachIntent.PRIVACY_CONCERN,
        OutreachIntent.BAD_EXPERIENCE,
        OutreachIntent.NOT_INTERESTED,
        OutreachIntent.TOO_BUSY,
    }:
        return ContentTier.OBJECTION
    if intent in REFUSAL_INTENTS and intent not in IMMEDIATE_EXIT_INTENTS:
        return ContentTier.OBJECTION
    return ContentTier.COMMAND


def _extract_first_json_object(raw: str) -> str:
    """Extract the first JSON object from a noisy LLM response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


# ── Heuristic pattern groups (module level so they compile once) ──────────────

_PARTIAL_OPT_OUT_PATTERNS = [
    r"\b(don't|do not|dont) call (me |us )?(at (work|the office|home)|during (work|the day|business hours)|in the (morning|evening|afternoon)|on (weekends?|sundays?|saturdays?)|before \w+|after \w+)",
    r"\b(only|just) (text|email|e-mail|message|write to) (me|us)\b",
    r"\b(text|email|e-mail) (me|us) instead\b",
    r"\b(i'd|i would|we'd|we would) (rather|prefer) (a )?(text|email|e-mail|letter)\b",
    r"\b(prefer|rather have) (a )?(text|texts|email|emails)\b",
]

_DNC_PATTERNS = [
    r"\bdo not call\b", r"\b(don't|dont) call (me|us|this number|here)\b", r"\b(don't|dont) call\b",
    r"\bstop calling\b", r"\bstop contacting\b", r"\bnever call\b",
    r"\bremove (me|us|my number|this number|our number)\b", r"\btake (me|us|my number|this number) off\b",
    r"\bremove from (your |the )?list\b", r"\bput (me|us) on (your |the )?do not call\b",
    r"\bno more (calls|reminders)\b", r"\b(don't|do not) want (any more|these|your) (calls|reminders)\b",
    r"\bunsubscribe\b", r"\bopt (me |us )?out\b", r"\blose my number\b",
    r"\bleave (me|us) alone\b", r"\bstop (bothering|harassing|disturbing) (me|us)\b",
    r"\bquit calling\b", r"\b(don't|dont|do not) (contact|phone) (me|us)\b",
]

_ATTORNEY_PATTERNS = [
    r"\b(my|our) (attorney|lawyer|legal counsel)\b",
    r"\b(talk|speak) (to|with) (my|our) (attorney|lawyer)\b",
    r"\brepresented by (an? )?(attorney|lawyer|counsel)\b",
    r"\bcease and desist\b",
]

_BANKRUPTCY_PATTERNS = [
    r"\bbankrupt(cy)?\b", r"\bchapter (7|seven|13|thirteen)\b",
]

_DECEASED_PATTERNS = [
    r"\bpassed away\b", r"\bdeceased\b",
    r"\b(he|she|they)( has| have)? (died|passed)\b",
    r"\b(he|she) is dead\b",
]

# First person only — harm aimed at the agent is ABUSE, not DISTRESS.
_DISTRESS_PATTERNS = [
    r"\b(kill|hurt|harm) myself\b", r"\bend (my|it all|my life)\b", r"\bsuicid(e|al)\b",
    r"\bwant to die\b", r"\bmedical emergency\b", r"\bheart attack\b",
    r"\b(can't|cannot) breathe\b", r"\bcall (an |the )?ambulance\b",
]

_WRONG_NUMBER_PATTERNS = [
    r"\bwrong number\b", r"\bwrong person\b",
    r"\bno one (here )?by that name\b", r"\bnobody (here )?by that name\b",
    r"\byou('ve| have) got the wrong\b",
    r"\bnever heard of (him|her|them|that person)\b",
    r"\b(he|she|they) (doesn't|does not|don't|do not) live here\b",
]

_LANGUAGE_PATTERNS = [
    r"\bi don't (speak|understand) (english|well)\b",
    r"\bno (habla|hablo) (ingles|english)\b",
    r"\bhabla(s)? (espanol|español)\b",
    r"\b(speak|in) spanish\b",
    r"\bmy english (is )?(not|isn't) (good|great|very)\b",
    r"\bsorry.{0,15}(english|language)\b",
]

_TRANSFER_PATTERNS = [
    r"\b(talk|speak|connect|transfer) (to |me to |with )?(a |an )?(real |actual |live )?(person|human|agent|representative|someone|somebody|operator|supervisor)\b",
    r"\blet me (talk|speak) (to|with) (a |your )?(real |actual |live )?(person|human|agent|representative|someone|somebody|supervisor|manager)\b",
    r"\bi (want|need|demand) (a |to talk to a |to speak to a )?(real |actual |live )?(person|human|agent|representative)\b",
    r"\bget me (a )?(real |actual |live )?(person|human|agent|representative)\b",
    r"\b(is there|can i get) (a )?(real |actual |live )?(person|human)\b",
    r"\bput (a |your )?(supervisor|manager|boss) on\b",
    r"\bi('d| would) (rather|prefer) (talk|speak) (to|with) (a )?(human|person)\b",
    r"\bstop (the )?script\b",
]

_HOLD_PATTERNS = [
    r"^(?:(?:yeah|yes|ok|okay|sure|um|uh)[,.]? )?(hold on|hang on|one (sec|second|moment|minute)|just a (sec|second|moment|minute)|give me a (sec|second|moment|minute)|wait a (sec|second|moment|minute)|bear with me)\b",
    r"\blet me (grab|get|check|find|look)\b",
]

_REPEAT_PATTERNS = [
    r"^(what|huh|sorry|pardon|excuse me)[?.! ]*$",
    r"\b(come again|say (that|it) again|repeat (that|it|yourself|the last part)|can you repeat|could you repeat)\b",
    r"\b(didn't|did not) (catch|hear|get|understand) (that|you|what you said)\b",
    r"\bwhat did you (just )?say\b",
]

_SLOW_DOWN_PATTERNS = [
    r"\bslow down\b", r"\b(speak|talk) (more )?slow(ly|er)\b", r"\b(talking|speaking) too fast\b", r"\btoo fast\b",
]

_GOODBYE_PATTERNS = [
    r"\b(good ?bye|bye|bye bye)\b", r"\b(have|got|gotta) (to )?go\b", r"\bhanging up\b",
]

_ALREADY_PAID_PATTERNS = [
    r"\b(already|just) (paid|made (the|a|that|my) payment|took care of (it|that|this)|sent (it|the payment|the money))\b",
    r"\b(i|we) (already )?paid (it|that|this|the bill|the balance|already|last|yesterday|online|in full)\b",
    r"\b(it|that|this|payment|the bill|the balance)('s been|'s| is| was| has been| was already) (already )?(paid|taken care of|settled)\b",
    r"\bpaid in full\b",
]

_DISPUTE_PATTERNS = [
    r"\b(don't|do not|doesn't|does not|never) owe\b",
    r"\b(wrong|incorrect) (amount|bill|charge|balance)\b",
    r"\b(amount|bill|charge|balance) (is|was) (wrong|incorrect|not right)\b",
    r"\bdispute\b",
    r"\bnot my (bill|charge|debt|balance)\b",
    r"\bi never (ordered|signed up|used|received) (that|this|it|the service)\b",
]

_HARDSHIP_PATTERNS = [
    r"\b(can't|cannot|can not) (afford|pay)\b",
    r"\b(lost my job|unemployed|out of work|no money|money is tight|tight on money)\b",
    r"\bfinancial(ly)? (hardship|trouble|difficult|struggling)\b",
    r"\bno budget\b", r"\btoo expensive\b",
]

_PAYMENT_PLAN_PATTERNS = [
    r"\bpayment (plan|arrangement)\b", r"\binstall?ments?\b",
    r"\bpay (it )?in (parts|pieces|installments)\b",
    r"\bsplit (it|the payment|the bill|the balance)\b",
    r"\bpartial payment\b", r"\bpay (a little|part of it|some of it)\b",
]

_PAY_NOW_PATTERNS = [
    r"\bpay (it |this |that |the bill )?(now|right now|today|over the phone)\b",
    r"\btake (my|a) (card|payment)\b", r"\bmy card number\b",
    r"\bcan i (just )?pay\b", r"\bhow (do|can) i pay\b",
]

_RESCHEDULE_PATTERNS = [
    r"\breschedule\b",
    r"\bmove (it|the appointment|my appointment|the date)\b",
    r"\bchange (the|my) (date|appointment|time|due date)\b",
    r"\b(a )?different (day|date|time)\b",
]

_CALLBACK_KEYWORDS = [
    "call back", "call me back", "call later", "try later", "another time", "not now", "call tomorrow",
    "call me tomorrow", "call next week", "call after", "call me after", "call in the",
    "not a good time", "bad time", "after hours", "we're closed", "we are closed",
    "office is closed",
]

_AI_PATTERNS = [
    r"\bare you (a |an )?(ai|bot|robot|machine|computer|artificial|recording|automated)",
    r"\bare you real\b",
    r"\bare you (a )?(real )?(human|person)\b",
    r"\bam i talking to (a |an )?(ai|bot|robot|machine|computer|real person|person|human|recording)",
    r"\bis this (a |an )?(ai|bot|robot|machine|automated|computer|recording|robocall)",
    r"\byou sound like (a |an )?(ai|bot|robot)",
    r"\byou('re| are) (a |an )?(ai|bot|robot)",
]

_TRUST_PATTERNS = [
    r"\b(is this|are you|this is|this sounds like) (a )?(scam|spam|fraud|phishing|legit|legitimate)\b",
    r"\bhow do i know (this|you)\b",
    r"\bsounds? (like a scam|too good|sketchy|fishy|suspicious)\b",
    r"\bi don't (trust|believe) (you|this)\b",
    r"\bwho are you (really|actually)\b",
    r"\bprove (it|you)\b",
]

_PRIVACY_PATTERNS = [
    r"\b(data|privacy|information) (protection|security|safe)\b",
    r"\bhow (do you|will you) (protect|handle|store|use) (my|our|the) (data|information)\b",
    r"\bprivacy (concern|policy|issue)\b",
    r"\bis (my|our) (data|information) (safe|secure)\b",
]

_ASK_SOURCE_PATTERNS = [
    r"\b(how|where) did you get (my|this|our) (number|contact|info|information)\b",
    r"\bwho gave you (my|this|our) (number|contact)\b",
    r"\bwhere is this number from\b",
]

_IDENTITY_DENIAL_PATTERNS = [
    r"\bthat('s| is) not me\b",
    r"\bi('m| am) not (dr\.?|doctor|him|her)\b",
    r"^no,? i('m| am) not\.?$",
    r"^no,? not me\.?$",
    r"^not me\.?$",
]

_THIRD_PARTY_PATTERNS = [
    r"\bthis is (his|her|their) (wife|husband|mother|mom|father|dad|son|daughter|partner|assistant|colleague|roommate|sister|brother)\b",
    r"\bi('m| am) (his|her|their) (wife|husband|mother|mom|father|dad|son|daughter|partner|assistant|colleague|roommate|sister|brother)\b",
    r"\b(he|she|they)('s| is| are) not (here|home|available|in)( right now| at the moment)?\b",
    r"\b(he|she|they)('s| is| are) (at work|out|away|busy)\b",
    r"\bcan i take (a )?message\b",
]

_GATEKEEPER_PATTERNS = [
    r"\bwho is this\b", r"\bwho's calling\b", r"\bwho('s| is) this\b",
    r"\bwhat is this (about|regarding)\b", r"\bwhat('s| is) this (about|regarding)\b",
    r"\b(i can|can i|let me) (put|get|connect) (him|her|them)\b",
]

_NDM_PATTERNS = [
    r"\bi('m| am) not the (right |one who |person who )?(decision maker|account holder)\b",
    r"\bi don't (handle|make) (those|that|the) (decision|bills?|payments?)\b",
    r"\bi('d| would)? need (to )?(check with|talk to|ask) (my|the) (husband|wife|partner|boss|manager|owner)\b",
    r"\byou('d| would) need to (talk|speak) to\b",
]

_EMAIL_KEYWORDS = ["email", "e-mail", "mail me", "send details", "send me the details", "send info", "in writing", "send me something"]

_TOO_BUSY_PATTERNS = [
    r"\btoo busy\b", r"\bbusy right now\b", r"\bno time\b",
    r"\b(i'm|i am|we're|we are) (in the middle of|tied up|swamped|slammed|driving)\b",
    r"\bcan't talk (right now|at the moment|now)\b",
    r"\b(not a good|bad) (time|moment)\b",
    r"\bkind of busy\b", r"\breally busy\b", r"\bsuper busy\b",
    r"\b(don't|do not) have (the )?time (for|right now|to)\b",
    r"\b(about to|heading into|going into|walking into) (a )?(meeting|appointment)\b",
    r"\b(only have|got) (a )?(second|minute|moment)\b",
]

_NOT_INTERESTED_PATTERNS = [
    r"\bno thanks\b", r"\bno thank you\b", r"\bnot interested\b",
    r"\bwe (do not|don't) want\b", r"\bnot needed\b",
    r"\b(i'll|we'll|i will|we will) pass\b",
    r"\bdon't bother\b", r"\bsave your (breath|time)\b",
    r"\bhard pass\b", r"\bnot for (us|me)\b",
]

_BAD_EXPERIENCE_PATTERNS = [
    r"\b(bad|terrible|horrible|awful|negative|worst|poor) (experience|service)\b",
    r"\bgot (burned|scammed|ripped off)\b",
    r"\b(i want to|i'd like to) (file a |make a )?complain",
]

_CANNOT_PROCEED_PATTERNS = [
    r"\bnot (allowed|permitted) to (discuss|talk about|give out)\b",
    r"\b(against|violates?) (our |the |company )?policy\b",
    r"\bpolicy (doesn't|does not|won't|will not) (allow|permit|let)\b",
]

_CLARIFY_NAME_PATTERNS = [
    r"\b(how do you spell|can you spell|spell it|what was the name|how do you pronounce|did you say)\b",
]

_META_PATTERNS = [
    r"\b(responses?|answers?|replies?) (are )?(too )?(short|brief|vague)\b",
    r"\btalk more\b", r"\belaborate\b", r"\bmore detail\b",
    r"\bexplain (me )?more\b", r"\bgive me more\b",
]

_INTEREST_PATTERNS = [
    r"\binterested\b", r"\bokay\b", r"\bok\b", r"\bsure\b", r"\bgo ahead\b", r"\btell me more\b",
]


def _heuristic_intent(transcript: str) -> tuple[OutreachIntent, bool, bool, ContentTier]:
    """Deterministic classification. Returns (intent, is_exit, counts_as_refusal, tier).

    Ordered by priority: safety + compliance first, then conversation control,
    then reminder outcomes, then objections. Anything unmatched is UNCLEAR.
    """
    text = _normalize_text(transcript)
    if not text:
        return OutreachIntent.UNCLEAR, False, False, DEFAULT_TIER
    words = _word_count(text)

    I = OutreachIntent
    # ── P1 safety (checked before anything else) ──
    if _any(_DISTRESS_PATTERNS, text):
        return I.DISTRESS, True, False, ContentTier.COMMAND

    # ── P0 compliance hard-stops ──
    if _any(_ATTORNEY_PATTERNS, text):
        return I.ATTORNEY, True, False, ContentTier.COMMAND
    if _any(_BANKRUPTCY_PATTERNS, text):
        return I.BANKRUPTCY, True, False, ContentTier.COMMAND
    if _any(_DECEASED_PATTERNS, text):
        return I.DECEASED, True, False, ContentTier.COMMAND
    # Partial opt-outs ("don't call me at work") must be checked before full DNC.
    if _any(_PARTIAL_OPT_OUT_PATTERNS, text):
        return I.PARTIAL_OPT_OUT, False, False, ContentTier.COMMAND
    if _any(_DNC_PATTERNS, text):
        return I.DO_NOT_CALL, True, True, ContentTier.COMMAND
    if _any(_WRONG_NUMBER_PATTERNS, text):
        return I.WRONG_NUMBER, True, True, ContentTier.COMMAND
    if _any(_LANGUAGE_PATTERNS, text):
        return I.LANGUAGE_BARRIER, True, True, ContentTier.COMMAND

    # Abuse aimed at NORA (see app/services/abuse.py for the three tiers).
    # Venting about their own situation ("shit, I forgot to pay") is not abuse.
    if is_abusive(text):
        return I.ABUSE, False, False, ContentTier.OBJECTION

    # Past this point swearing is just emphasis, so it must not hide the answer.
    text = strip_profanity(text)

    # ── Escalation request ──
    if _any(_TRANSFER_PATTERNS, text):
        return I.TRANSFER_TO_HUMAN, False, False, ContentTier.COMMAND

    # ── Conversation control (short utterances only, to avoid false hits) ──
    if words <= 8 and _any(_HOLD_PATTERNS, text):
        return I.HOLD_REQUEST, False, False, ContentTier.COMMAND
    if words <= 8 and _any(_REPEAT_PATTERNS, text):
        return I.REPEAT_REQUEST, False, False, ContentTier.COMMAND
    if words <= 10 and _any(_SLOW_DOWN_PATTERNS, text):
        return I.SLOW_DOWN, False, False, ContentTier.COMMAND

    # ── Reminder outcomes ──
    if _any(_ALREADY_PAID_PATTERNS, text):
        return I.ALREADY_PAID, False, False, ContentTier.COMMAND
    if _any(_DISPUTE_PATTERNS, text):
        return I.DISPUTE, False, False, ContentTier.OBJECTION
    if _any(_PAYMENT_PLAN_PATTERNS, text):
        return I.PAYMENT_PLAN, False, False, ContentTier.COMMAND
    if _any(_HARDSHIP_PATTERNS, text):
        return I.HARDSHIP, False, False, ContentTier.OBJECTION
    if _any(_PAY_NOW_PATTERNS, text):
        return I.PAY_NOW, False, False, ContentTier.COMMAND
    if _any(_RESCHEDULE_PATTERNS, text):
        return I.RESCHEDULE, False, False, ContentTier.COMMAND

    if words <= 8 and _any(_GOODBYE_PATTERNS, text):
        return I.GOODBYE, False, False, ContentTier.COMMAND

    # ── Trust / identity ──
    if _any(_AI_PATTERNS, text):
        return I.ASK_IF_AI, False, False, ContentTier.DEFINITION
    if _any(_TRUST_PATTERNS, text):
        return I.TRUST_CONCERN, False, False, ContentTier.OBJECTION
    if _any(_PRIVACY_PATTERNS, text):
        return I.PRIVACY_CONCERN, False, False, ContentTier.OBJECTION
    if _any(_ASK_SOURCE_PATTERNS, text):
        return I.ASK_SOURCE, False, False, ContentTier.DEFINITION
    if _any(_IDENTITY_DENIAL_PATTERNS, text):
        return I.IDENTITY_DENIAL, False, False, ContentTier.COMMAND
    if _any(_THIRD_PARTY_PATTERNS, text):
        return I.THIRD_PARTY, False, False, ContentTier.COMMAND
    if _any(_NDM_PATTERNS, text):
        return I.NOT_DECISION_MAKER, False, False, ContentTier.COMMAND
    if _any(_GATEKEEPER_PATTERNS, text):
        return I.GATEKEEPER, False, False, ContentTier.COMMAND

    # ── Logistics ──
    if any(k in text for k in _EMAIL_KEYWORDS):
        return I.EMAIL_REQUEST, False, False, ContentTier.COMMAND
    if any(k in text for k in _CALLBACK_KEYWORDS):
        return I.CALL_BACK_LATER, False, False, ContentTier.COMMAND

    # Social check-in replies are not objections and never count as refusals.
    if _is_social_smalltalk(text):
        return I.UNCLEAR, False, False, ContentTier.COMMAND

    # ── Name clarification ──
    if _any(_CLARIFY_NAME_PATTERNS, text):
        return I.CLARIFY_NAME, False, False, ContentTier.COMMAND

    # ── Soft objections ──
    if _any(_TOO_BUSY_PATTERNS, text):
        return I.TOO_BUSY, False, True, ContentTier.OBJECTION
    if _any(_BAD_EXPERIENCE_PATTERNS, text):
        return I.BAD_EXPERIENCE, False, True, ContentTier.OBJECTION
    if _any(_CANNOT_PROCEED_PATTERNS, text):
        return I.CANNOT_PROCEED, False, True, ContentTier.COMMAND
    if _any(_META_PATTERNS, text):
        return I.ASK_MORE_INFO, False, False, ContentTier.DEFINITION
    if _any(_NOT_INTERESTED_PATTERNS, text):
        return I.NOT_INTERESTED, False, True, ContentTier.OBJECTION

    if _any(_INTEREST_PATTERNS, text):
        return I.INTERESTED, False, False, ContentTier.COMMAND

    if "?" in text or any(k in text for k in ["can you explain", "more info", "details", "what is this about"]):
        return I.ASK_MORE_INFO, False, False, ContentTier.DEFINITION

    return I.UNCLEAR, False, False, DEFAULT_TIER


def _should_prefer_heuristic_intent(
    model_intent: OutreachIntent,
    heuristic_intent: OutreachIntent,
    heuristic_refusal: bool,
) -> bool:
    if heuristic_intent == OutreachIntent.UNCLEAR:
        return False

    if model_intent == OutreachIntent.UNCLEAR:
        return True

    if model_intent == heuristic_intent:
        return False

    if heuristic_refusal and model_intent not in REFUSAL_INTENTS:
        return True

    if heuristic_intent in IMMEDIATE_EXIT_INTENTS:
        return True

    if heuristic_intent in HIGH_CONFIDENCE_HEURISTIC_INTENTS and model_intent in WEAKER_MODEL_INTENTS:
        return True

    return False


# ─── System prompt: forces JSON-only output ───
# Only intents the classifier may emit (legacy billing-audit intents excluded).
_CLASSIFIER_INTENTS = [
    "ALREADY_PAID", "DISPUTE", "HARDSHIP", "PAYMENT_PLAN", "PAY_NOW", "RESCHEDULE",
    "CALL_BACK_LATER", "EMAIL_REQUEST", "PARTIAL_OPT_OUT", "HOLD_REQUEST", "REPEAT_REQUEST",
    "SLOW_DOWN", "GOODBYE", "GATEKEEPER", "THIRD_PARTY", "IDENTITY_DENIAL", "NOT_DECISION_MAKER",
    "INTERESTED", "NOT_INTERESTED", "TOO_BUSY", "BAD_EXPERIENCE", "CANNOT_PROCEED",
    "ASK_MORE_INFO", "ASK_SOURCE", "ASK_IF_AI", "TRUST_CONCERN", "PRIVACY_CONCERN",
    "TRANSFER_TO_HUMAN", "DO_NOT_CALL", "WRONG_NUMBER", "LANGUAGE_BARRIER", "ATTORNEY",
    "BANKRUPTCY", "DECEASED", "DISTRESS", "ABUSE", "CLARIFY_NAME", "FAKE_NAME", "UNCLEAR",
]
_ALL_INTENTS = "|".join(_CLASSIFIER_INTENTS)
_CLASSIFIABLE = {OutreachIntent(name) for name in _CLASSIFIER_INTENTS}

_CLASSIFIER_SYSTEM_PROMPT = f"""\
You are a fast intent classifier for an outbound payment and service reminder call.
The caller (NORA) reminds a customer about a payment or service on their account.
Classify the customer's latest reply. Return ONLY a JSON object in this exact shape:
{{"intent": "<{_ALL_INTENTS}>", "tier": "<COMMAND|DEFINITION|EXPLANATION|OBJECTION>"}}

Intent rules:
- ALREADY_PAID: says the payment was already made or the bill is settled.
- DISPUTE: says they don't owe it, the amount/charge is wrong, or wants to dispute.
- HARDSHIP: can't afford to pay, lost job, financial difficulty.
- PAYMENT_PLAN: asks to pay in parts, installments, or a payment arrangement.
- PAY_NOW: wants to pay now / over the phone / asks how to pay.
- RESCHEDULE: wants to move an appointment, service, or due date.
- CALL_BACK_LATER: asks to be called another time, or gives a time to call back.
- EMAIL_REQUEST: wants details by email, text, or in writing.
- PARTIAL_OPT_OUT: limits contact without stopping it ("don't call me at work", "text only").
- HOLD_REQUEST: asks NORA to wait a moment ("hold on", "one sec").
- REPEAT_REQUEST: didn't hear or asks NORA to repeat.
- SLOW_DOWN: asks NORA to speak slower.
- GOODBYE: ends the conversation ("bye", "I have to go") without refusing anything specific.
- GATEKEEPER: screening the call ("who is this?", "what's this regarding?").
- THIRD_PARTY: someone other than the account holder answered, or the account holder is not available.
- IDENTITY_DENIAL: says they are not the person NORA asked for.
- NOT_DECISION_MAKER: says someone else handles the bills or account.
- INTERESTED: cooperative or agreeing ("okay", "sure", "go ahead").
- NOT_INTERESTED: flat refusal to engage ("not interested", "no thanks").
- TOO_BUSY: can't talk right now because busy.
- BAD_EXPERIENCE: complains about past service or experience.
- CANNOT_PROCEED: says policy or rules stop them from discussing it.
- ASK_MORE_INFO: asks what the call is about or for more details.
- ASK_SOURCE: asks how NORA got their number.
- ASK_IF_AI: asks if they are talking to an AI, bot, robot, or recording.
- TRUST_CONCERN: suspects a scam or asks NORA to prove legitimacy.
- PRIVACY_CONCERN: worried about data privacy or security.
- TRANSFER_TO_HUMAN: asks for a real person, agent, or supervisor.
- DO_NOT_CALL: asks to stop calling, be removed, or opt out entirely. COMPLIANCE CRITICAL.
- WRONG_NUMBER: says it's the wrong number or nobody by that name lives there.
- LANGUAGE_BARRIER: can't communicate in English or asks for another language.
- ATTORNEY: says they have a lawyer/attorney or sends a cease and desist.
- BANKRUPTCY: mentions bankruptcy.
- DECEASED: says the account holder has died.
- DISTRESS: mentions self-harm, a medical emergency, or immediate danger.
- ABUSE: insults, profanity aimed at NORA, or threats.
- CLARIFY_NAME: asks how a name is spelled or pronounced.
- FAKE_NAME: gives an obviously fake or joke name.
- UNCLEAR: a plain answer (yes/no, a date, a number, a name) or nothing else fits.

Tier rules:
- COMMAND: short turns, answers, logistics, compliance.
- DEFINITION: what-is questions, brief clarification.
- EXPLANATION: how-it-works questions.
- OBJECTION: pushback, complaints, disputes, hardship, trust or privacy concerns.

Return ONLY JSON. No markdown, no backticks, no explanation."""


def _result(intent: OutreachIntent, tier: ContentTier, turn_count: int, t_start: float) -> IntentResult:
    """Build an IntentResult with exit/refusal flags derived from the intent itself."""
    max_words, max_tokens = get_tier_limits(tier, turn_count)
    return IntentResult(
        is_exit=intent in IMMEDIATE_EXIT_INTENTS,
        primary_intent=intent,
        tier=tier,
        counts_as_refusal=intent in REFUSAL_INTENTS,
        max_words=max_words,
        max_tokens=max_tokens,
        latency_ms=(time.time() - t_start) * 1000,
    )


async def classify_intent(transcript: str, turn_count: int = 0) -> IntentResult:
    """
    Classify the customer's intent.

    Returns an IntentResult with:
      - is_exit: whether the call must end (compliance / safety)
      - primary_intent: the conversational intent for call handling
      - tier: COMMAND / DEFINITION / EXPLANATION / OBJECTION
      - counts_as_refusal: whether this turn increments the refusal counter
      - max_words / max_tokens: limits for the main LLM response
      - latency_ms: how long the classification took
    """
    t_start = time.time()
    heuristic_intent, _heuristic_exit, heuristic_refusal, heuristic_tier = _heuristic_intent(transcript)
    normalized_text = _normalize_text(transcript)

    # ── Fast path 1: deterministic heuristics (compliance phrases always land here) ──
    if heuristic_intent in HIGH_CONFIDENCE_HEURISTIC_INTENTS:
        result = _result(heuristic_intent, heuristic_tier, turn_count, t_start)
        print(f"[INTENT] Fast-path heuristic: {heuristic_intent.value} ({result.latency_ms:.0f}ms)")
        return result

    # ── Fast path 2: plain numeric / date answers are answers, not intents ──
    if _looks_like_numeric_answer(normalized_text) or _is_social_smalltalk(normalized_text):
        return _result(OutreachIntent.UNCLEAR, ContentTier.COMMAND, turn_count, t_start)

    # Low-information heuristic matches (e.g. CLARIFY_NAME) still skip the model.
    if heuristic_intent != OutreachIntent.UNCLEAR:
        return _result(heuristic_intent, heuristic_tier, turn_count, t_start)

    try:
        response = await _client.chat.completions.create(
            model=_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,
            max_tokens=40,
            stream=False,
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(_extract_first_json_object(raw))

        intent_str = str(result.get("intent", "UNCLEAR")).upper()
        tier_str = str(result.get("tier", "COMMAND")).upper()

        try:
            primary_intent = OutreachIntent(intent_str)
        except ValueError:
            primary_intent = OutreachIntent.UNCLEAR
        if primary_intent not in _CLASSIFIABLE:
            primary_intent = OutreachIntent.UNCLEAR

        try:
            tier = ContentTier(tier_str)
        except ValueError:
            tier = DEFAULT_TIER

        if _should_prefer_heuristic_intent(primary_intent, heuristic_intent, heuristic_refusal):
            primary_intent = heuristic_intent

        if primary_intent == OutreachIntent.END_CALL:
            primary_intent = OutreachIntent.GOODBYE

        # Guardrail: the model may only escalate to a compliance hard-stop when
        # the transcript actually contains supporting words; otherwise a
        # misclassification would end a normal call.
        if primary_intent in IMMEDIATE_EXIT_INTENTS | {OutreachIntent.ABUSE}:
            evidence = {
                OutreachIntent.DO_NOT_CALL: ["call", "remove", "stop", "list", "opt", "contact"],
                OutreachIntent.WRONG_NUMBER: ["wrong", "number", "name", "live here", "who"],
                OutreachIntent.LANGUAGE_BARRIER: ["english", "spanish", "speak", "understand", "habla"],
                OutreachIntent.ATTORNEY: ["attorney", "lawyer", "counsel", "legal", "cease"],
                OutreachIntent.BANKRUPTCY: ["bankrupt", "chapter"],
                OutreachIntent.DECEASED: ["passed", "died", "dead", "deceased", "funeral"],
                OutreachIntent.DISTRESS: ["hurt", "kill", "die", "emergency", "breathe", "ambulance", "suicid", "help"],
                OutreachIntent.ABUSE: [],
            }[primary_intent]
            if primary_intent == OutreachIntent.ABUSE:
                if not is_abusive(normalized_text):
                    primary_intent = OutreachIntent.UNCLEAR
            elif not any(w in normalized_text for w in evidence):
                primary_intent = OutreachIntent.UNCLEAR

        if primary_intent != OutreachIntent.UNCLEAR:
            tier = _intent_to_default_tier(primary_intent) if tier == DEFAULT_TIER else tier
        else:
            tier = ContentTier.COMMAND

        return _result(primary_intent, tier, turn_count, t_start)

    except Exception as e:
        latency_ms = (time.time() - t_start) * 1000
        print(f"[INTENT] Classification failed ({latency_ms:.0f}ms): {e}")
        return _result(heuristic_intent, heuristic_tier, turn_count, t_start)
