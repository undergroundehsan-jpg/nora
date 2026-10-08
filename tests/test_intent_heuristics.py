"""Deterministic intent heuristics (no network: every case hits the regex fast path)."""
import asyncio

import pytest

from app.services.intent import (
    IMMEDIATE_EXIT_INTENTS,
    REFUSAL_INTENTS,
    OutreachIntent as I,
    _heuristic_intent,
    classify_intent,
)


@pytest.mark.parametrize(
    "transcript, expected",
    [
        # P1 safety / P0 compliance
        ("I want to kill myself", I.DISTRESS),
        ("please stop calling me", I.DO_NOT_CALL),
        ("Thanks, but take me off your list", I.DO_NOT_CALL),
        ("remove my number", I.DO_NOT_CALL),
        ("don't call me at work", I.PARTIAL_OPT_OUT),
        ("just text me instead", I.PARTIAL_OPT_OUT),
        ("you have the wrong number", I.WRONG_NUMBER),
        ("nobody here by that name", I.WRONG_NUMBER),
        ("talk to my lawyer", I.ATTORNEY),
        ("I filed for bankruptcy", I.BANKRUPTCY),
        ("she passed away last month", I.DECEASED),
        ("sorry my english is not good", I.LANGUAGE_BARRIER),
        ("screw you", I.ABUSE),
        # Harm aimed at the agent is abuse, not caller distress (found on a live call).
        ("You should kill yourself.", I.ABUSE),
        ("go die", I.ABUSE),
        ("drop dead", I.ABUSE),
        ("you are useless", I.ABUSE),
        ("I am going to hurt myself", I.DISTRESS),
        # Escalation
        ("let me talk to a real person", I.TRANSFER_TO_HUMAN),
        # Conversation control
        ("hold on one sec", I.HOLD_REQUEST),
        ("one second", I.HOLD_REQUEST),
        ("What?", I.REPEAT_REQUEST),
        ("sorry can you repeat that", I.REPEAT_REQUEST),
        ("slow down please", I.SLOW_DOWN),
        ("okay bye", I.GOODBYE),
        # Reminder outcomes
        ("I already paid that last week", I.ALREADY_PAID),
        ("it's been paid", I.ALREADY_PAID),
        ("I don't owe you anything", I.DISPUTE),
        ("that's the wrong amount", I.DISPUTE),
        ("I can't afford it right now", I.HARDSHIP),
        ("can I set up a payment plan", I.PAYMENT_PLAN),
        ("can I pay over the phone", I.PAY_NOW),
        ("I need to reschedule", I.RESCHEDULE),
        ("call me back tomorrow", I.CALL_BACK_LATER),
        # Trust / identity
        ("are you a robot", I.ASK_IF_AI),
        ("is this a scam", I.TRUST_CONCERN),
        ("how did you get my number", I.ASK_SOURCE),
        ("this is his wife", I.THIRD_PARTY),
        ("she's not home right now", I.THIRD_PARTY),
        ("who is this?", I.GATEKEEPER),
        # Soft objections
        ("I'm too busy right now", I.TOO_BUSY),
        ("not interested", I.NOT_INTERESTED),
        # Plain answers are answers, not refusals
        ("no", I.UNCLEAR),
        ("yes", I.UNCLEAR),
        ("I'm good, how are you?", I.UNCLEAR),
    ],
)
def test_heuristic_intent(transcript, expected):
    assert _heuristic_intent(transcript)[0] == expected


def test_bare_no_is_not_a_refusal():
    intent, _exit, refusal, _tier = _heuristic_intent("no")
    assert intent == I.UNCLEAR and refusal is False


def test_goodbye_is_not_a_refusal():
    intent, _exit, refusal, _tier = _heuristic_intent("bye")
    assert intent == I.GOODBYE and refusal is False


def test_long_sentence_mentioning_bye_is_not_goodbye():
    intent = _heuristic_intent("my son said bye to the old account years ago and we switched everything over")[0]
    assert intent != I.GOODBYE


def test_compliance_intents_exit_the_call():
    for intent in (I.DO_NOT_CALL, I.WRONG_NUMBER, I.ATTORNEY, I.BANKRUPTCY, I.DECEASED, I.DISTRESS):
        assert intent in IMMEDIATE_EXIT_INTENTS


@pytest.mark.parametrize(
    "transcript, expected",
    [
        # Hard stops must win over the numeric/answer shortcuts.
        ("my account is active but stop calling", I.DO_NOT_CALL),
        ("one second", I.HOLD_REQUEST),
        ("I already paid", I.ALREADY_PAID),
    ],
)
def test_classify_intent_fast_path_precedence(transcript, expected):
    result = asyncio.run(classify_intent(transcript, turn_count=2))
    assert result.primary_intent == expected
    assert result.is_exit == (expected in IMMEDIATE_EXIT_INTENTS)
    assert result.counts_as_refusal == (expected in REFUSAL_INTENTS)


def test_numeric_answer_is_unclear_not_refusal():
    result = asyncio.run(classify_intent("twenty five", turn_count=2))
    assert result.primary_intent == I.UNCLEAR
    assert result.counts_as_refusal is False



def test_harm_to_agent_is_abuse_not_distress():
    """A live call classified "You should kill yourself." as UNCLEAR and the model
    answered as if the caller were in distress. Abuse and distress must not mix."""
    assert _heuristic_intent("you should kill yourself")[0] == I.ABUSE
    assert _heuristic_intent("kill yourself")[0] == I.ABUSE
    assert _heuristic_intent("i want to kill myself")[0] == I.DISTRESS
    assert _heuristic_intent("i might hurt myself tonight")[0] == I.DISTRESS
