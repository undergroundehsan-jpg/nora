"""Abuse detection: what counts, what doesn't, and how the call ends.

Three tiers (app/services/abuse.py):
  severe   → the call ends immediately, no offer
  directed → one calm offer, then the call ends
  venting  → not abuse at all; the conversation continues
"""
import copy

import pytest

from app.services import policy
from app.services.abuse import is_abusive, is_severe, strip_profanity
from app.services.intent import OutreachIntent, _heuristic_intent
from app.session import SAMPLE_LEAD_CONTEXT, Session

SEVERE = [
    "kill yourself", "you should kill yourself", "I said kill yourself",
    "go die", "just die", "drop dead", "you should die", "you should rot",
    "I hope you die", "I hope you rot", "I hope you suffer",
    "I'll kill you", "I will hurt you", "I am going to find you",
    "I'll come to your office", "watch your back", "you're dead",
]

DIRECTED = [
    # Profanity aimed at the agent
    "fuck you", "Fuck off!", "fuck yourself", "f*** you", "f**k you", "f u c k you",
    "s*** off", "screw you", "piss off", "bugger off", "get lost", "shove it",
    "shut up", "shut the hell up", "shut the fuck up", "stfu", "wtf", "gtfo",
    "go to hell", "damn you",
    # Insults
    "you're an idiot", "you are a moron", "you idiot", "you fool", "you liar",
    "you're useless", "you are pathetic", "you people are liars", "you suck",
    "stupid bot", "useless robot", "stupid machine", "damn recording",
    # Standalone profanity about the call
    "this is bullshit", "what a load of bollocks", "bitch", "asshole",
    # Romanised Urdu / Hindi
    "bhenchod", "behenchod", "madarchod", "chutiya", "gandu", "harami",
    "kameena", "besharam", "bakwas", "kutta",
]

NOT_ABUSE = [
    # Venting about their own situation
    "oh shit, I forgot to pay", "shit I thought I paid that", "damn, I forgot",
    "I already fucking paid it", "crap", "damn", "ugh",
    "I am too damn busy right now", "just fucking call me tomorrow",
    # Ordinary replies
    "yes speaking", "I already paid", "call me tomorrow", "not interested",
    "this is a bad time", "who is this?", "are you a robot",
    # Distress is about the caller, never abuse
    "I want to kill myself", "I might hurt myself tonight",
]


@pytest.mark.parametrize("utterance", SEVERE)
def test_severe_abuse_is_detected(utterance):
    assert is_severe(utterance), utterance
    assert is_abusive(utterance), utterance
    assert _heuristic_intent(utterance)[0] == OutreachIntent.ABUSE


@pytest.mark.parametrize("utterance", DIRECTED)
def test_directed_abuse_is_detected(utterance):
    assert is_abusive(utterance), utterance
    assert _heuristic_intent(utterance)[0] == OutreachIntent.ABUSE


@pytest.mark.parametrize("utterance", DIRECTED)
def test_directed_abuse_is_not_treated_as_a_threat(utterance):
    """Only threats end the call instantly; an insult gets one calm offer."""
    assert not is_severe(utterance), utterance


@pytest.mark.parametrize("utterance", NOT_ABUSE)
def test_ordinary_speech_is_not_abuse(utterance):
    assert not is_abusive(utterance), utterance
    assert _heuristic_intent(utterance)[0] != OutreachIntent.ABUSE


@pytest.mark.parametrize("utterance", ["I want to kill myself", "I might hurt myself"])
def test_self_harm_is_distress_not_abuse(utterance):
    assert _heuristic_intent(utterance)[0] == OutreachIntent.DISTRESS


@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("I already fucking paid", OutreachIntent.ALREADY_PAID),
        ("I am too damn busy", OutreachIntent.TOO_BUSY),
        ("just fucking call me tomorrow", OutreachIntent.CALL_BACK_LATER),
    ],
)
def test_swearing_does_not_hide_the_real_answer(utterance, expected):
    assert _heuristic_intent(utterance)[0] == expected


def test_strip_profanity_keeps_the_sentence_readable():
    assert strip_profanity("I already fucking paid") == "I already paid"
    assert strip_profanity("this is total bullshit") == "this is total"


@pytest.mark.parametrize("utterance", ["leave me alone", "stop bothering me", "quit calling"])
def test_requests_to_be_left_alone_are_opt_outs(utterance):
    assert _heuristic_intent(utterance)[0] == OutreachIntent.DO_NOT_CALL


# ── How the call behaves ────────────────────────────────────────────────────

def _session():
    s = Session(websocket=None, session_id="abuse")
    s.lead_context = copy.deepcopy(SAMPLE_LEAD_CONTEXT)
    s.turn_count = 2
    return s


@pytest.mark.parametrize("utterance", SEVERE[:6])
def test_threats_end_the_call_without_an_offer(utterance):
    action = policy.decide(_session(), utterance, "ABUSE")
    assert action.kind == policy.END
    assert action.disposition == "ENDED_ABUSE"
    assert "callback" not in action.text.lower()


@pytest.mark.parametrize("utterance", ["fuck you", "you're an idiot", "this is bullshit"])
def test_insults_get_one_offer_then_the_call_ends(utterance):
    session = _session()
    first = policy.decide(session, utterance, "ABUSE")
    assert first.kind == policy.SAY
    second = policy.decide(session, utterance, "ABUSE")
    assert second.kind == policy.END and second.disposition == "ENDED_ABUSE"


def test_extra_terms_can_be_added_without_code_changes(tmp_path, monkeypatch):
    import app.services.abuse as abuse_mod
    terms = tmp_path / "abuse_terms.json"
    terms.write_text('{"directed": ["\\\\bnonsense word\\\\b"]}', encoding="utf-8")
    monkeypatch.setattr(abuse_mod, "_EXTRA_TERMS_PATH", terms)
    monkeypatch.setattr(abuse_mod, "_DIRECTED_RE",
                        abuse_mod._compile(abuse_mod._DIRECTED_PATTERNS, "directed"))
    assert abuse_mod.is_abusive("that is a nonsense word")
