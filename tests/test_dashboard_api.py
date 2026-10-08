"""Dashboard read API: aggregation from the call logs and lead store."""
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services import analytics
from app.services import leads as leads_mod

LEADS = [
    {"lead_id": "L-1", "lead_name": "Ayesha Siddiqui", "practice_name": "Northside Clinic",
     "phone": "+923325026869", "campaign": "Clinic Billing", "amount_due": "$85.00",
     "due_date": "14 October 2026", "email": "a@example.com"},
    {"lead_id": "L-2", "lead_name": "Bilal Hussain", "practice_name": "Northside Clinic",
     "phone": "+923111222333", "campaign": "Utility Due"},
]

SESSIONS = [
    {"type": "session", "session_id": "S-PAID", "lead_id": "L-1",
     "start_time": "2026-10-07T09:00:00Z", "end_time": "2026-10-07T09:01:30Z",
     "total_turns": 4, "last_intent": "ALREADY_PAID", "final_outcome": "ALREADY_PAID",
     "identity_verified": True, "mood_trajectory": ["neutral", "positive"]},
    {"type": "session", "session_id": "S-DNC", "lead_id": "L-2",
     "start_time": "2026-10-07T10:00:00Z", "end_time": "2026-10-07T10:00:40Z",
     "total_turns": 2, "last_intent": "DO_NOT_CALL", "final_outcome": "DO_NOT_CALL",
     "identity_verified": False, "mood_trajectory": ["cold"]},
]

TURNS = [
    {"type": "turn", "session_id": "S-PAID", "lead_id": "L-1", "turn_index": 1,
     "user_transcript": "yes speaking", "llm_response": "Thanks for confirming.", "intent": "UNCLEAR"},
    {"type": "turn", "session_id": "S-PAID", "lead_id": "L-1", "turn_index": 2,
     "user_transcript": "I already paid", "llm_response": "Thanks for letting me know.", "intent": "ALREADY_PAID"},
    {"type": "latency", "session_id": "S-PAID", "turn_index": 1,
     "latency_ms": {"intent": 300, "tts_first_audio": 1100}},
]

EVENTS = [
    {"session_id": "S-DNC", "lead_id": "L-2", "timestamp": "2026-10-07T10:00:35Z",
     "lead_name": "Bilal Hussain", "phone": "+923111222333", "type": "suppression", "reason": "DO_NOT_CALL"},
    {"session_id": "S-PAID", "lead_id": "L-1", "timestamp": "2026-10-07T09:01:00Z",
     "lead_name": "Ayesha Siddiqui", "phone": "+923325026869", "type": "callback_request",
     "reason": "DISPUTE", "requested_time": "friday at 3"},
]

ATTEMPTS = [
    {"timestamp": "2026-10-07T09:00:00Z", "call_sid": "S-PAID", "lead_id": "L-1",
     "to": "+923325026869", "status": "completed", "duration": 90},
    {"timestamp": "2026-10-07T11:00:00Z", "call_sid": "CA-NOANSWER", "lead_id": "L-1",
     "to": "+923325026869", "status": "no-answer", "duration": 0},
]


@pytest.fixture
def api(tmp_path, monkeypatch):
    def write(name, rows):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    monkeypatch.setattr(analytics, "SESSIONS_LOG", write("sessions.jsonl", SESSIONS))
    monkeypatch.setattr(analytics, "TURNS_LOG", write("turns.jsonl", TURNS))
    monkeypatch.setattr(analytics, "EVENTS_LOG", write("events.jsonl", EVENTS))
    monkeypatch.setattr(analytics, "ATTEMPTS_LOG", write("attempts.jsonl", ATTEMPTS))
    analytics.clear_cache()

    leads_path = tmp_path / "leads.json"
    leads_path.write_text(json.dumps(LEADS), encoding="utf-8")
    monkeypatch.setenv("LEADS_FILE", str(leads_path))
    leads_mod._cache.update({"mtime": None, "path": None, "leads": {}})

    return TestClient(main.app)


# ── Calls ───────────────────────────────────────────────────────────────────

def test_calls_join_sessions_leads_and_attempts(api):
    calls = api.get("/api/dashboard/calls").json()["calls"]
    paid = next(c for c in calls if c["sessionId"] == "S-PAID")
    assert paid["customer"] == "Ayesha Siddiqui"
    assert paid["phone"] == "+923325026869"
    assert paid["campaign"] == "Clinic Billing"
    assert paid["status"] == "Success"
    assert paid["intent"] == "Payment Confirmed"
    assert paid["sentiment"] == "Cooperative"
    assert paid["duration"] == "1m 30s"      # from the Twilio attempt record
    assert paid["retries"] == 1              # a second attempt went unanswered


def test_unanswered_calls_appear_even_without_a_session(api):
    calls = api.get("/api/dashboard/calls").json()["calls"]
    missed = next(c for c in calls if c["sessionId"] == "CA-NOANSWER")
    assert missed["status"] == "No Answer" and missed["turns"] == 0


def test_calls_can_be_filtered_and_searched(api):
    assert all(c["status"] == "Success" for c in
               api.get("/api/dashboard/calls?status=Success").json()["calls"])
    found = api.get("/api/dashboard/calls?q=bilal").json()["calls"]
    assert found and all("Bilal" in c["customer"] for c in found)


def test_call_detail_returns_the_conversation_in_order(api):
    detail = api.get("/api/dashboard/calls/S-PAID").json()["call"]
    roles = [line["role"] for line in detail["transcript"]]
    assert roles == ["Customer", "Nora", "Customer", "Nora"]
    assert detail["transcript"][0]["text"] == "yes speaking"


def test_unknown_call_id_is_reported_not_crashed(api):
    assert api.get("/api/dashboard/calls/NOPE").json()["call"] is None


@pytest.mark.parametrize(
    "disposition, expected",
    [
        ("ALREADY_PAID", "Success"), ("CALLBACK_SCHEDULED", "Rescheduled"),
        ("DO_NOT_CALL", "Opt-Out"), ("WRONG_NUMBER", "Wrong Number"),
        ("NO_RESPONSE", "No Answer"), ("VOICEMAIL_LEFT", "No Answer"),
        ("TECH_TROUBLE", "Failed"), ("SAFETY_ESCALATION", "Escalated"),
        ("SUCCESS_MEETING", "Success"),       # legacy rows still map
        (None, "Declined"),
    ],
)
def test_disposition_maps_to_a_dashboard_status(disposition, expected):
    assert analytics.status_for(disposition, total_turns=3) == expected


# ── Overview ────────────────────────────────────────────────────────────────

def test_summary_counts_outcomes_over_the_window(api):
    s = api.get("/api/dashboard/summary?days=90").json()
    assert s["totalCalls"] == 3           # two sessions + one unanswered attempt
    assert s["resolved"] == 1
    assert s["optOuts"] == 1
    assert s["outcomes"]["noAnswer"] == 1
    assert len(s["volume"]) == 90
    assert s["avgHandleSeconds"] > 0


def test_summary_reports_today_separately(api):
    s = api.get("/api/dashboard/summary?days=90").json()
    assert s["totalCallsToday"] == 0      # the fixture data is from 7 October
    assert s["totalCallsAllTime"] == 3


# ── Customers ───────────────────────────────────────────────────────────────

def test_customers_merge_leads_with_call_history(api):
    rows = {c["id"]: c for c in api.get("/api/dashboard/customers").json()["customers"]}
    assert rows["L-1"]["name"] == "Ayesha Siddiqui"
    assert rows["L-1"]["amountDue"] == 85.0
    assert rows["L-1"]["status"] == "Overdue"
    assert rows["L-1"]["attempts"] == 2
    assert rows["L-1"]["lastContacted"] == "Oct 07, 2026"


def test_customer_on_the_do_not_call_list_is_flagged(api):
    rows = {c["id"]: c for c in api.get("/api/dashboard/customers").json()["customers"]}
    assert rows["L-2"]["status"] == "DNC"


# ── Compliance and escalations ──────────────────────────────────────────────

def test_dnc_list_comes_from_suppression_events(api):
    entries = api.get("/api/dashboard/compliance/dnc").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["phone"] == "+923111222333"
    assert entries[0]["source"] == "Customer opt-out"
    assert entries[0]["addedBy"] == "NORA (automated)"


def test_audit_log_is_empty_until_pre_dial_checks_exist(api):
    assert api.get("/api/dashboard/compliance/audit").json()["events"] == []


def test_escalations_are_built_from_events(api):
    rows = api.get("/api/dashboard/escalations").json()["escalations"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "Billing dispute"
    assert rows[0]["priority"] == "Medium"
    assert rows[0]["customer"] == "Ayesha Siddiqui"
    assert rows[0]["requestedTime"] == "friday at 3"


# ── Analytics ───────────────────────────────────────────────────────────────

def test_analytics_series_are_computed_from_real_calls(api):
    a = api.get("/api/dashboard/analytics?days=90").json()
    assert a["totals"]["calls"] == 3
    assert {o["name"] for o in a["outcomes"]} == {"Success", "Opt-Out", "No Answer"}
    assert any(h["calls"] for h in a["hourly"])
    assert a["latencyMs"]["tts_first_audio"] == 1100
    assert {c["campaign"] for c in a["campaigns"]} == {"Clinic Billing", "Utility Due"}


def test_campaigns_are_derived_from_leads(api):
    rows = {c["name"]: c for c in api.get("/api/dashboard/campaigns").json()["campaigns"]}
    assert rows["Clinic Billing"]["total"] == 1
    assert rows["Utility Due"]["total"] == 1


def test_empty_logs_do_not_break_any_endpoint(tmp_path, monkeypatch):
    for attr in ("SESSIONS_LOG", "TURNS_LOG", "EVENTS_LOG", "ATTEMPTS_LOG"):
        monkeypatch.setattr(analytics, attr, tmp_path / "missing.jsonl")
    analytics.clear_cache()
    client = TestClient(main.app)
    for path in ("/api/dashboard/summary", "/api/dashboard/calls", "/api/dashboard/analytics",
                 "/api/dashboard/customers", "/api/dashboard/escalations",
                 "/api/dashboard/compliance/dnc", "/api/dashboard/compliance/audit",
                 "/api/dashboard/campaigns"):
        assert client.get(path).status_code == 200, path


# ── Labels for the dashboard (raised by a real call showing "Failed / Unclear") ──

def test_ending_an_abusive_call_is_not_reported_as_a_failure():
    assert analytics.status_for("ENDED_ABUSE", total_turns=3) == "Abuse"
    assert analytics.status_for("TECH_TROUBLE", total_turns=3) == "Failed"


@pytest.mark.parametrize(
    "intent, label",
    [
        ("ABUSE", "Abuse"),
        ("DISTRESS", "Safety Concern"),
        ("NOT_INTERESTED", "Not Interested"),
        ("THIRD_PARTY", "Wrong Person"),
        ("GOODBYE", "Caller Ended"),
        ("ALREADY_PAID", "Payment Confirmed"),
    ],
)
def test_intents_have_readable_labels(intent, label):
    assert analytics.INTENT_LABELS[intent] == label


def test_call_dialled_without_a_lead_is_matched_back_by_phone(api, tmp_path, monkeypatch):
    """A call placed straight from the dialler showed as 'Unknown' on the dashboard."""
    sessions = tmp_path / "s2.jsonl"
    sessions.write_text(json.dumps({
        "type": "session", "session_id": "S-NOLEAD", "lead_id": None,
        "start_time": "2026-10-08T20:05:00Z", "end_time": "2026-10-08T20:06:10Z",
        "total_turns": 3, "last_intent": "ABUSE", "final_outcome": "ENDED_ABUSE"}) + "\n", encoding="utf-8")
    attempts = tmp_path / "a2.jsonl"
    attempts.write_text(json.dumps({
        "timestamp": "2026-10-08T20:05:00Z", "call_sid": "S-NOLEAD", "lead_id": None,
        "to": "+923325026869", "status": "completed", "duration": 70}) + "\n", encoding="utf-8")
    monkeypatch.setattr(analytics, "SESSIONS_LOG", sessions)
    monkeypatch.setattr(analytics, "ATTEMPTS_LOG", attempts)
    analytics.clear_cache()

    call = api.get("/api/dashboard/calls").json()["calls"][0]
    assert call["customer"] == "Ayesha Siddiqui"   # matched by the number dialled
    assert call["status"] == "Abuse"
    assert call["intent"] == "Abuse"
