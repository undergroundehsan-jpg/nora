"""System prompt: builds for every intent/stage, stays compact, no legacy or contradictory rules."""
import copy

import pytest

from app.services.intent import OutreachIntent
from app.session import SAMPLE_LEAD_CONTEXT
from app.utils.language import build_system_prompt

LEAD = copy.deepcopy(SAMPLE_LEAD_CONTEXT)
STAGES = ["INTRO", "VERIFY", "REMIND", "RESOLVE", "CLOSE"]


@pytest.mark.parametrize("intent", [i.value for i in OutreachIntent] + ["WRAP_UP_TIMEOUT"])
@pytest.mark.parametrize("stage", STAGES)
def test_prompt_builds_for_every_intent_and_stage(intent, stage):
    prompt = build_system_prompt(
        "en", max_words=40, primary_intent=intent, lead_context=LEAD, turn_count=3,
        stage=stage, identity_verified=stage not in {"INTRO", "VERIFY"},
    )
    assert f"stage={stage}" in prompt
    assert "LIMIT: 40 words max" in prompt


def test_fixed_prefix_is_identical_across_turns():
    a = build_system_prompt("en", primary_intent="UNCLEAR", lead_context=LEAD, stage="REMIND", turn_count=2)
    b = build_system_prompt("en", primary_intent="TRUST_CONCERN", lead_context=LEAD, stage="RESOLVE", turn_count=5,
                            filler_used=True, recent_openers=["Just to confirm"])
    prefix_a = a.split("CALL STATE:")[0]
    prefix_b = b.split("CALL STATE:")[0]
    assert prefix_a == prefix_b
    assert len(prefix_a) > 500


def test_prompt_requires_honest_ai_disclosure():
    prompt = build_system_prompt("en", primary_intent="ASK_IF_AI", lead_context=LEAD, stage="RESOLVE")
    assert "automated assistant" in prompt
    lowered = prompt.lower()
    assert "never reveal you are ai" not in lowered
    assert "real person" not in lowered
    assert "deny" not in lowered


def test_no_duplicate_or_legacy_sections():
    prompt = build_system_prompt("en", primary_intent="UNCLEAR", lead_context=LEAD, stage="REMIND")
    assert prompt.count("HOW YOU SPEAK") == 1
    for legacy in ("YOUR VOICE", "eligibility", "NPI", "billing audit", "SOCIAL PROOF", "pitch"):
        assert legacy not in prompt


def test_prompt_is_much_smaller_than_before():
    prompt = build_system_prompt("en", primary_intent="NOT_INTERESTED", lead_context=LEAD, stage="RESOLVE",
                                 refusal_count=2, turn_count=6, filler_used=True)
    # The previous builder produced roughly 12-14k characters for this case.
    assert len(prompt) < 5000


def test_account_details_are_gated_on_identity():
    prompt = build_system_prompt("en", primary_intent="GATEKEEPER", lead_context=LEAD, stage="VERIFY")
    assert "share only after the account holder is confirmed" in prompt
    assert "account holder confirmed=no" in prompt


def test_email_confirmation_instruction():
    prompt = build_system_prompt("en", primary_intent="EMAIL_REQUEST", lead_context=LEAD, stage="RESOLVE",
                                 email_pending_confirmation=True, email_address="a@b.com")
    assert "captured a@b.com" in prompt


def test_legacy_callers_without_stage_still_work():
    prompt = build_system_prompt("en", primary_intent="UNCLEAR", lead_context=None, turn_count=1)
    assert "stage=INTRO" in prompt
    assert "ACCOUNT INFO: none on file" in prompt
