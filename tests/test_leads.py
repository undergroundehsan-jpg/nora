"""Customer (lead) records: loading, validation, and propagation to the call."""
import json

import pytest

from app.services import leads as leads_mod
from app.services.leads import (
    LeadError,
    build_lead_context,
    lead_for_call,
    list_leads,
    load_lead,
    normalize_phone,
    phone_for_lead,
)
from app.services.twilio_bridge import _build_agent_ws_url, _build_inbound_webhook_url

SAMPLE = [
    {
        "lead_id": "T-1", "lead_name": "Ayesha Siddiqui", "practice_name": "Northside Clinic",
        "phone": "0092 332 5026869", "amount_due": "$85.00", "due_date": "14 October 2026",
        "member_id": "NC-991", "timezone": "Asia/Karachi",
    },
    {"lead_id": "T-2", "lead_name": "No Phone", "practice_name": "Northside Clinic", "phone": "12345"},
    {"no_id": True, "lead_name": "Ignored"},
]


@pytest.fixture
def leads_file(tmp_path, monkeypatch):
    path = tmp_path / "leads.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    monkeypatch.setenv("LEADS_FILE", str(path))
    leads_mod._cache.update({"mtime": None, "path": None, "leads": {}})
    return path


def test_loads_records_and_skips_ones_without_an_id(leads_file):
    assert sorted(l["lead_id"] for l in list_leads()) == ["T-1", "T-2"]


def test_lead_fields_reach_the_call_context(leads_file):
    lead = load_lead("T-1")
    assert lead["lead_name"] == "Ayesha Siddiqui"
    assert lead["amount_due"] == "$85.00" and lead["due_date"] == "14 October 2026"
    assert lead["call_type"] == "REMINDER"


def test_unknown_lead_is_an_error_not_a_silent_fallback(leads_file):
    assert load_lead("NOPE") is None
    with pytest.raises(LeadError):
        lead_for_call("NOPE")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+923325026869", "+923325026869"),
        ("+92 332 502 6869", "+923325026869"),
        ("0092 332 5026869", "+923325026869"),   # 00 is the international prefix
        ("92-332-5026869", "+923325026869"),
        ("0332 5026869", ""),                     # national form: country unknown
        ("12345", ""),                            # too short
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_unusable_phone_is_rejected(leads_file):
    assert phone_for_lead(load_lead("T-2")) == ""


def test_missing_optional_fields_are_simply_absent(leads_file):
    lead = load_lead("T-2")
    for field in ("amount_due", "due_date", "service", "appointment_time"):
        assert field not in lead


def test_file_edits_are_picked_up_without_restart(leads_file):
    assert load_lead("T-4") is None
    records = json.loads(leads_file.read_text(encoding="utf-8"))
    records.append({"lead_id": "T-4", "lead_name": "Added Later", "phone": "+923325026869"})
    leads_file.write_text(json.dumps(records), encoding="utf-8")
    import os
    os.utime(leads_file, (0, 0))  # force a different mtime
    assert load_lead("T-4")["lead_name"] == "Added Later"


def test_sample_customer_only_when_explicitly_allowed(leads_file, monkeypatch):
    monkeypatch.setenv("ALLOW_SAMPLE_LEAD", "true")
    assert lead_for_call("")["lead_id"] == "SAMPLE"
    monkeypatch.setenv("ALLOW_SAMPLE_LEAD", "false")
    with pytest.raises(LeadError):
        lead_for_call("")


def test_prompt_uses_the_lead_details(leads_file):
    from app.utils.language import build_system_prompt
    prompt = build_system_prompt("en", primary_intent="UNCLEAR", lead_context=load_lead("T-1"),
                                 stage="REMIND", identity_verified=True)
    assert "$85.00" in prompt and "14 October 2026" in prompt
    assert "Ayesha Siddiqui" in prompt and "Northside Clinic" in prompt


def test_prompt_says_nothing_is_on_file_when_the_lead_has_no_amount(leads_file):
    from app.utils.language import build_system_prompt
    prompt = build_system_prompt("en", primary_intent="UNCLEAR", lead_context=load_lead("T-2"),
                                 stage="REMIND", identity_verified=True)
    assert "No amount, due date, or service is on file" in prompt


def test_greeting_uses_the_lead_name(leads_file):
    from app.utils.language import build_greeting_variants
    _name, text = build_greeting_variants(load_lead("T-1")["lead_name"],
                                          load_lead("T-1")["practice_name"])[0]
    assert "Ayesha Siddiqui" in text and "Northside Clinic" in text


# ── The lead id has to survive every hop of a phone call ────────────────────

def test_lead_id_is_attached_to_the_twiml_webhook(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok-free.app")
    assert "lead_id=L-1001" in _build_inbound_webhook_url("L-1001")
    assert "lead_id" not in _build_inbound_webhook_url("")


def test_lead_id_is_passed_to_the_agent_socket():
    url = _build_agent_ws_url("sess-1", "L-1001")
    assert "session_id=sess-1" in url and "lead_id=L-1001" in url
    assert "lead_id" not in _build_agent_ws_url("sess-1", "")


def test_build_lead_context_masks_nothing_but_keeps_the_id():
    ctx = build_lead_context({"lead_id": " L-9 ", "lead_name": "A", "phone": "+923325026869"})
    assert ctx["lead_id"] == "L-9" and ctx["phone"] == "+923325026869"
