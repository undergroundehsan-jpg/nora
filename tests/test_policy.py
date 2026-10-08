"""Policy engine: every edge case and escalation flow, no audio or network."""
import copy
import time

import pytest

from app.services import policy
from app.session import SAMPLE_LEAD_CONTEXT, Session

GREETING = (
    "Hi, is this Dr. Sarah Mitchell? This is NORA calling from Mitchell Family Clinic "
    "with a payment or service reminder. Is now a good time?"
)


@pytest.fixture
def session():
    s = Session(websocket=None, session_id="test")
    s.lead_context = copy.deepcopy(SAMPLE_LEAD_CONTEXT)
    s.chat_history.append({"role": "assistant", "content": GREETING})
    s.turn_count = 1
    return s


def say(session, text, intent):
    return policy.decide(session, text, intent)


# ── P1 / P0 ──────────────────────────────────────────────────────────────────

def test_distress_ends_with_crisis_line(session):
    action = say(session, "I want to hurt myself", "DISTRESS")
    assert action.kind == policy.END
    assert "9 8 8" in action.text and "9 1 1" in action.text
    assert action.disposition == "SAFETY_ESCALATION"
    assert action.records == [{"type": "escalation", "reason": "DISTRESS"}]


@pytest.mark.parametrize(
    "intent, disposition",
    [
        ("DO_NOT_CALL", "DO_NOT_CALL"),
        ("WRONG_NUMBER", "WRONG_NUMBER"),
        ("ATTORNEY", "ATTORNEY_REPRESENTED"),
        ("BANKRUPTCY", "BANKRUPTCY"),
        ("DECEASED", "DECEASED"),
        ("LANGUAGE_BARRIER", "LANGUAGE_BARRIER"),
    ],
)
def test_compliance_stops_end_without_llm(session, intent, disposition):
    action = say(session, "…", intent)
    assert action.kind == policy.END
    assert action.disposition == disposition
    assert "?" not in action.text  # no follow-up question on a hard stop


def test_do_not_call_is_recorded_for_suppression(session):
    action = say(session, "stop calling", "DO_NOT_CALL")
    assert {"type": "suppression", "reason": "DO_NOT_CALL"} in action.records


def test_abuse_deescalates_once_then_ends(session):
    first = say(session, "screw you", "ABUSE")
    assert first.kind == policy.SAY
    second = say(session, "screw you again", "ABUSE")
    assert second.kind == policy.END and second.disposition == "ENDED_ABUSE"


# ── Identity / right party ───────────────────────────────────────────────────

def test_confirmation_after_greeting_verifies_identity(session):
    action = say(session, "yes speaking", "UNCLEAR")
    assert action.kind == policy.LLM
    assert session.identity_verified is True
    assert session.call_stage == "REMIND"


def test_reminder_marked_delivered_after_remind_turn(session):
    session.identity_verified = True
    policy.note_assistant_turn(session, "REMIND")
    assert session.reminder_delivered is True
    assert session.call_stage == "RESOLVE"


def test_third_party_gets_callback_without_disclosure(session):
    action = say(session, "this is his wife", "THIRD_PARTY")
    assert action.kind == policy.SAY
    assert "account" not in action.text.lower()  # no account details to a third party
    assert session.identity_verified is False
    done = say(session, "try tomorrow afternoon", "UNCLEAR")
    assert done.kind == policy.END
    assert done.disposition == "CALLBACK_SCHEDULED"
    assert "tomorrow afternoon" in done.text
    assert done.records[0]["requested_time"] == "tomorrow afternoon"


# ── P2 escalation → callback ─────────────────────────────────────────────────

@pytest.mark.parametrize("intent", ["TRANSFER_TO_HUMAN", "DISPUTE", "HARDSHIP", "PAYMENT_PLAN", "BAD_EXPERIENCE"])
def test_escalations_offer_callback(session, intent):
    action = say(session, "…", intent)
    assert action.kind == policy.SAY
    assert session.pending_callback_reason == intent
    assert session.escalation_reason == intent


def test_escalation_with_time_in_same_sentence_confirms_immediately(session):
    action = say(session, "I can't afford it, call me friday after 3pm", "HARDSHIP")
    assert action.kind == policy.END
    assert action.disposition == "CALLBACK_SCHEDULED"
    assert "friday" in action.text and "3pm" in action.text


def test_escalation_declined(session):
    say(session, "that's the wrong amount", "DISPUTE")
    action = say(session, "no forget it", "UNCLEAR")
    assert action.kind == policy.END
    assert action.disposition == "ESCALATION_DECLINED"


def test_pay_now_never_takes_payment_and_caps_callback_asks(session):
    first = say(session, "can I pay over the phone", "PAY_NOW")
    assert first.kind == policy.SAY and "can't take payment" in first.text
    assert say(session, "yes", "UNCLEAR").text == policy.template("ASK_TIME")
    assert say(session, "sure", "UNCLEAR").kind == policy.SAY
    final = say(session, "okay", "INTERESTED")
    assert final.kind == policy.END
    assert final.disposition == "CALLBACK_NO_TIME"


def test_new_question_while_pending_callback_goes_to_llm(session):
    say(session, "...", "TRANSFER_TO_HUMAN")
    action = say(session, "what is this about?", "ASK_MORE_INFO")
    assert action.kind == policy.LLM
    assert session.pending_callback_reason is None


# ── P4 soft objections ───────────────────────────────────────────────────────

def test_busy_gets_one_offer_then_close(session):
    first = say(session, "I'm busy", "TOO_BUSY")
    assert first.kind == policy.SAY
    second = say(session, "I said I'm busy", "TOO_BUSY")
    assert second.kind == policy.END and second.disposition == "DECLINED"


def test_busy_with_time_schedules_callback(session):
    action = say(session, "I'm busy, call me in two hours", "TOO_BUSY")
    assert action.kind == policy.END
    assert "in two hours" in action.text


def test_not_interested_then_stop_is_do_not_call(session):
    say(session, "not interested", "NOT_INTERESTED")
    action = say(session, "just stop them", "UNCLEAR")
    assert action.kind == policy.END
    assert action.disposition == "DO_NOT_CALL"


def test_not_interested_then_no_closes_softly(session):
    say(session, "not interested", "NOT_INTERESTED")
    action = say(session, "no", "UNCLEAR")
    assert action.kind == policy.END and action.disposition == "DECLINED"


# ── Outcomes ─────────────────────────────────────────────────────────────────

def test_already_paid_closes(session):
    action = say(session, "I already paid", "ALREADY_PAID")
    assert action.kind == policy.END
    assert session.outcome == "PAID"


def test_partial_opt_out_records_preference(session):
    action = say(session, "don't call me at work", "PARTIAL_OPT_OUT")
    assert action.kind == policy.END
    assert action.records[0]["type"] == "preference"
    assert session.preferences["contact_preference"] == "don't call me at work"


def test_reschedule_asks_for_time_then_confirms(session):
    assert say(session, "I need to reschedule", "RESCHEDULE").kind == policy.SAY
    action = say(session, "next tuesday morning", "UNCLEAR")
    assert action.kind == policy.END
    assert "next tuesday morning" in action.text


# ── Conversation control ─────────────────────────────────────────────────────

def test_hold_checkin_and_timeout(session):
    action = say(session, "hold on", "HOLD_REQUEST")
    assert action.kind == policy.SAY and session.hold_active
    start = session.hold_started_at
    assert policy.hold_status(session, start + 5) is None
    assert policy.hold_status(session, start + policy.HOLD_CHECKIN_SECS + 1) == "CHECKIN"
    session.hold_checkin_sent = True
    assert policy.hold_status(session, start + policy.HOLD_CHECKIN_SECS + 2) is None
    assert policy.hold_status(session, start + policy.HOLD_MAX_SECS + 1) == "TIMEOUT"


def test_any_reply_ends_hold(session):
    say(session, "hold on", "HOLD_REQUEST")
    say(session, "okay I'm back", "UNCLEAR")
    assert session.hold_active is False


def test_repeat_replays_last_reply_then_offers_callback(session):
    session.last_response = "Has the payment already been taken care of?"
    first = say(session, "what?", "REPEAT_REQUEST")
    assert first.kind == policy.REPEAT and first.text == session.last_response
    say(session, "what?", "REPEAT_REQUEST")
    third = say(session, "what?", "REPEAT_REQUEST")
    assert third.kind == policy.SAY and session.pending_callback_reason == "REPEAT_LIMIT"


def test_slow_down_lowers_speed_and_repeats(session):
    session.last_response = "Has it been paid?"
    action = say(session, "slow down", "SLOW_DOWN")
    assert action.kind == policy.REPEAT
    assert action.text.endswith("Has it been paid?")
    assert session.tts_time_scale == pytest.approx(1.2)


def test_ai_question_is_answered_honestly(session):
    first = say(session, "are you a robot", "ASK_IF_AI")
    assert first.kind == policy.SAY and "automated assistant" in first.text
    assert session.pending_callback_reason is None
    second = say(session, "seriously, a robot?", "ASK_IF_AI")
    assert "automated call" in second.text
    assert session.pending_callback_reason == "AI_DISCLOSURE"


def test_goodbye_ends(session):
    action = say(session, "bye", "GOODBYE")
    assert action.kind == policy.END and action.disposition == "CALLER_ENDED"


def test_loop_breaker_after_same_question_twice(session):
    q = "Has the payment for your account already been taken care of?"
    session.chat_history += [
        {"role": "user", "content": "hmm"},
        {"role": "assistant", "content": q},
        {"role": "user", "content": "uh"},
        {"role": "assistant", "content": q},
    ]
    action = say(session, "maybe", "UNCLEAR")
    assert action.kind == policy.SAY
    assert session.pending_callback_reason == "LOOP"


def test_normal_turn_goes_to_llm(session):
    session.identity_verified = True
    assert say(session, "what's this about?", "ASK_MORE_INFO").kind == policy.LLM


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, expected",
    [
        ("after 5 pm", "after 5 pm"),
        ("call me next tuesday morning", "next tuesday morning"),
        ("in two hours", "in two hours"),
        ("tomorrow at 3", "tomorrow at 3"),
        ("no idea", None),
        ("yes", None),
    ],
)
def test_extract_callback_time(text, expected):
    assert policy.extract_callback_time(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Please leave a message after the tone", "VOICEMAIL"),
        ("The person you are calling is not available to take your call", "VOICEMAIL"),
        ("For billing press 1", "IVR"),
        ("Hello?", None),
    ],
)
def test_classify_telephony_prompt(text, expected):
    assert policy.classify_telephony_prompt(text) == expected


def test_all_templates_format():
    for name in policy._TEMPLATES:
        policy.template(name, org="Test Org", time="tomorrow")


# ── Follow-up accuracy ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, expected",
    [
        ("no, not tomorrow, friday works", "friday"),
        ("not tomorrow", None),
        ("good morning", None),
        ("at one point we paid", None),
        ("around five pm", "five pm"),
        ("after 5:30", "after 5:30"),
        ("anytime is fine", policy.ANY_TIME),
        ("whenever", policy.ANY_TIME),
    ],
)
def test_extract_callback_time_edge_cases(text, expected):
    assert policy.extract_callback_time(text) == expected


def test_anytime_closes_with_business_hours_promise(session):
    action = say(session, "I'm busy, call me anytime", "TOO_BUSY")
    assert action.kind == policy.END
    assert action.disposition == "CALLBACK_NO_TIME"
    assert action.records[0]["requested_time"] is None


def test_new_outcome_while_waiting_for_time_wins(session):
    say(session, "I'm busy", "TOO_BUSY")
    action = say(session, "actually I already paid on tuesday", "ALREADY_PAID")
    assert action.kind == policy.END
    assert action.disposition == "ALREADY_PAID"
    assert session.callback_time is None


def test_repeat_keeps_the_callback_question_open(session):
    first = say(session, "I need to reschedule", "RESCHEDULE")
    session.last_response = first.text
    replay = say(session, "what?", "REPEAT_REQUEST")
    assert replay.kind == policy.REPEAT and replay.text == first.text
    assert session.pending_callback_reason == "RESCHEDULE"
    done = say(session, "thursday afternoon", "UNCLEAR")
    assert done.kind == policy.END and "thursday afternoon" in done.text


def test_hold_keeps_the_callback_question_open(session):
    say(session, "call me later", "CALL_BACK_LATER")
    say(session, "hold on", "HOLD_REQUEST")
    assert session.pending_callback_reason == "CALL_BACK_LATER"
    done = say(session, "ok, tomorrow at 10 am", "UNCLEAR")
    assert done.kind == policy.END and "tomorrow at 10 am" in done.text


def test_declining_callback_after_ai_disclosure_continues_the_call(session):
    say(session, "are you a robot", "ASK_IF_AI")
    say(session, "a robot?", "ASK_IF_AI")
    action = say(session, "no, just keep going", "UNCLEAR")
    assert action.kind == policy.LLM
    assert session.pending_callback_reason is None


def test_third_party_who_does_not_know_gets_polite_close(session):
    say(session, "this is his wife", "THIRD_PARTY")
    ask = say(session, "hmm", "UNCLEAR")
    assert ask.text == policy.template("ASK_TIME_THIRD_PARTY")
    action = say(session, "I don't know", "UNCLEAR")
    assert action.kind == policy.END
    assert action.disposition == "CALLBACK_NO_TIME"


def test_still_busy_without_time_closes(session):
    say(session, "I'm busy", "TOO_BUSY")
    action = say(session, "I'm still busy", "TOO_BUSY")
    assert action.kind == policy.END and action.disposition == "DECLINED"


def test_abuse_then_end_request_closes(session):
    say(session, "screw you", "ABUSE")
    action = say(session, "just end the call", "UNCLEAR")
    assert action.kind == policy.END


def test_good_time_question_is_not_identity_confirmation(session):
    session.chat_history.append({"role": "assistant", "content": "Is this a good time to talk?"})
    say(session, "yes", "UNCLEAR")
    assert session.identity_verified is False


def test_yes_with_screening_question_still_confirms_identity(session):
    say(session, "yes, who's calling?", "GATEKEEPER")
    assert session.identity_verified is True


def test_okay_alone_does_not_confirm_identity(session):
    say(session, "okay", "INTERESTED")
    assert session.identity_verified is False


def test_reminder_not_marked_delivered_without_reminder_content(session):
    session.identity_verified = True
    policy.note_assistant_turn(session, "REMIND", "Sure, who am I speaking with today?")
    assert session.reminder_delivered is False
    policy.note_assistant_turn(session, "REMIND", "Just a reminder that your payment is coming due. Has it been taken care of?")
    assert session.reminder_delivered is True


_END_TEMPLATES = [
    "DISTRESS", "DO_NOT_CALL", "WRONG_NUMBER", "ATTORNEY", "BANKRUPTCY", "DECEASED", "LANGUAGE_BARRIER",
    "ABUSE_END", "SOFT_CLOSE", "CANNOT_PROCEED", "ALREADY_PAID", "CALLBACK_CONFIRMED", "CALLBACK_NO_TIME",
    "CALLBACK_DECLINED", "PARTIAL_OPT_OUT", "HOLD_TIMEOUT", "GOODBYE", "TECH_TROUBLE", "VOICEMAIL",
]


@pytest.mark.parametrize("name", _END_TEMPLATES)
def test_every_call_ending_line_says_goodbye(name):
    text = policy.template(name, org="Test Org", time="tomorrow").lower()
    assert any(p in text for p in ("goodbye", "good day", "great day", "take care", "thank you"))
    assert not text.rstrip().endswith("?")


# ── Abuse severity (from a live call) ───────────────────────────────────────

@pytest.mark.parametrize(
    "utterance",
    ["I said kill yourself.", "go die", "drop dead", "I hope you rot", "I'll kill you"],
)
def test_threats_end_the_call_immediately(session, utterance):
    """Someone threatening the agent gets no callback offer, just a short close."""
    action = say(session, utterance, "ABUSE")
    assert action.kind == policy.END
    assert action.disposition == "ENDED_ABUSE"
    assert "callback" not in action.text.lower()


def test_mild_abuse_still_gets_one_calm_offer(session):
    action = say(session, "this is bullshit", "ABUSE")
    assert action.kind == policy.SAY
    assert session.abuse_count == 1
    second = say(session, "seriously, this is rubbish", "ABUSE")
    assert second.kind == policy.END and second.disposition == "ENDED_ABUSE"


def test_severe_abuse_detection_is_second_person_only():
    assert policy.is_severe_abuse("you should kill yourself") is True
    assert policy.is_severe_abuse("i want to kill myself") is False
