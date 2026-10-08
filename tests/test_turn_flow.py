"""Glue between main.py's speculation guess, the classifier, and session state."""
import pytest

from app.main import _predict_speculative_intent_and_tier
from app.services.intent import OutreachIntent, _heuristic_intent
from app.session import Session


@pytest.mark.parametrize(
    "transcript",
    [
        "can you email me the details",
        "is this a scam",
        "who is this?",
        "what is this regarding",
        "stop calling me",
        "I already paid",
        "call me tomorrow",
        "hold on",
        "yes",
    ],
)
def test_speculation_matches_classifier_or_abstains(transcript):
    predicted, tier = _predict_speculative_intent_and_tier(transcript)
    if predicted == "UNCLEAR":
        return
    heuristic, _exit, _refusal, heuristic_tier = _heuristic_intent(transcript)
    assert heuristic.value == predicted
    assert heuristic_tier.value == tier


@pytest.mark.parametrize(
    "transcript",
    ["stop calling me", "I already paid", "call me tomorrow", "hold on", "I'm too busy", "are you a bot"],
)
def test_no_speculation_for_policy_handled_turns(transcript):
    assert _predict_speculative_intent_and_tier(transcript)[0] == "UNCLEAR"


def test_session_stage_progression_and_latency_marks():
    s = Session(websocket=None, session_id="t")
    s.turn_count = 1
    assert s.call_stage == "INTRO"
    s.turn_count = 2
    assert s.call_stage == "VERIFY"
    s.identity_verified = True
    assert s.call_stage == "REMIND"
    s.reminder_delivered = True
    assert s.call_stage == "RESOLVE"
    s.outcome = "PAID"
    assert s.call_stage == "CLOSE"

    s.turn_metrics = {}
    s.mark_turn("speech_end")
    s.turn_metrics["speech_end"] -= 0.5
    s.mark_turn("intent")
    first = s.turn_metrics["intent"]
    s.mark_turn("intent")  # first write wins
    assert s.turn_metrics["intent"] == first
    assert 450 <= s.turn_latency_ms()["intent"] <= 700


def test_intents_used_by_session_modes_exist():
    for name in ("THIRD_PARTY", "DISPUTE", "HARDSHIP", "RESCHEDULE", "PAY_NOW", "ALREADY_PAID", "ATTORNEY"):
        OutreachIntent(name)
