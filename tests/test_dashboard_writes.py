"""Dashboard write API: customers, do-not-call list, escalations, dialling."""
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services import analytics, stores
from app.services import leads as leads_mod

LEADS = [{"lead_id": "L-1", "lead_name": "Ayesha Siddiqui", "phone": "+923325026869",
          "practice_name": "Northside Clinic", "campaign": "Clinic Billing"}]


@pytest.fixture
def api(tmp_path, monkeypatch):
    leads_path = tmp_path / "leads.json"
    leads_path.write_text(json.dumps(LEADS), encoding="utf-8")
    monkeypatch.setenv("LEADS_FILE", str(leads_path))
    leads_mod._cache.update({"mtime": None, "path": None, "leads": {}})

    monkeypatch.setattr(stores, "DNC_PATH", tmp_path / "dnc.json")
    monkeypatch.setattr(stores, "ESCALATION_STATE_PATH", tmp_path / "escalation_state.json")
    for attr in ("SESSIONS_LOG", "TURNS_LOG", "EVENTS_LOG", "ATTEMPTS_LOG"):
        monkeypatch.setattr(analytics, attr, tmp_path / f"{attr.lower()}.jsonl")
    analytics.clear_cache()
    return TestClient(main.app)


# ── Customers ───────────────────────────────────────────────────────────────

def test_add_customer_appears_in_the_list(api):
    r = api.post("/api/dashboard/customers", json={
        "name": "Bilal Hussain", "phone": "+923111222333",
        "email": "b@example.com", "campaign": "Utility Due", "amount_due": "PKR 4,200",
    })
    assert r.status_code == 200 and r.json()["ok"]
    created = r.json()["customer"]
    assert created["lead_id"].startswith("L-")

    names = [c["name"] for c in api.get("/api/dashboard/customers").json()["customers"]]
    assert "Bilal Hussain" in names


def test_new_customer_can_be_called_immediately(api):
    lead_id = api.post("/api/dashboard/customers", json={
        "name": "Bilal Hussain", "phone": "0092 311 1222333"}).json()["customer"]["lead_id"]
    from app.services.leads import load_lead
    assert load_lead(lead_id)["phone"] == "+923111222333"   # normalised on the way in


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"phone": "+923111222333"}, "name is required"),
        ({"name": "No Phone"}, "international format"),
        ({"name": "Bad Phone", "phone": "0332 1234567"}, "international format"),
    ],
)
def test_invalid_customers_are_rejected_with_a_reason(api, payload, expected):
    r = api.post("/api/dashboard/customers", json=payload)
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_duplicate_phone_is_rejected(api):
    r = api.post("/api/dashboard/customers", json={"name": "Copy", "phone": "+923325026869"})
    assert r.status_code == 400 and "already exists" in r.json()["error"]


def test_edit_and_delete_customer(api):
    r = api.patch("/api/dashboard/customers/L-1", json={"campaign": "Utility Due", "email": "new@example.com"})
    assert r.json()["customer"]["campaign"] == "Utility Due"
    assert api.delete("/api/dashboard/customers/L-1").json()["ok"]
    assert api.get("/api/dashboard/customers").json()["customers"] == []


def test_editing_an_unknown_customer_is_a_404(api):
    assert api.patch("/api/dashboard/customers/NOPE", json={"campaign": "X"}).status_code == 404


# ── CSV import ──────────────────────────────────────────────────────────────

def test_csv_import_reports_successes_and_failures(api):
    r = api.post("/api/dashboard/customers/import", json={"rows": [
        {"name": "Good One", "phone": "+923001112222", "campaign": "Utility Due"},
        {"name": "Missing Phone"},
        {"name": "Bad Phone", "phone": "123"},
        {"name": "Also Good", "phone": "+923004445555"},
    ]})
    body = r.json()
    assert body["imported"] == 2 and body["failed"] == 2
    assert body["errors"][0]["row"] == 2
    assert len(api.get("/api/dashboard/customers").json()["customers"]) == 3  # 1 existing + 2 new


# ── Do-not-call list ────────────────────────────────────────────────────────

def test_manual_dnc_entry_is_added_and_listed(api):
    r = api.post("/api/dashboard/compliance/dnc", json={
        "phone": "+923111222333", "name": "Bilal", "note": "Asked by email"})
    assert r.json()["ok"]
    entries = api.get("/api/dashboard/compliance/dnc").json()["entries"]
    assert entries[0]["phone"] == "+923111222333"
    assert entries[0]["source"] == "Manual entry"
    assert entries[0]["removable"] is True


def test_manual_entry_can_be_removed(api):
    entry = api.post("/api/dashboard/compliance/dnc", json={"phone": "+923111222333"}).json()["entry"]
    assert api.delete(f"/api/dashboard/compliance/dnc/{entry['id']}").json()["ok"]
    assert api.get("/api/dashboard/compliance/dnc").json()["entries"] == []


def test_duplicate_dnc_number_is_rejected(api):
    api.post("/api/dashboard/compliance/dnc", json={"phone": "+923111222333"})
    r = api.post("/api/dashboard/compliance/dnc", json={"phone": "+92 311 1222333"})
    assert r.status_code == 400 and "already on the do-not-call list" in r.json()["error"]


def test_opt_out_recorded_on_a_call_cannot_be_removed(api, tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({
        "session_id": "S-1", "lead_id": "L-1", "timestamp": "2026-10-07T10:00:00Z",
        "lead_name": "Ayesha Siddiqui", "phone": "+923325026869",
        "type": "suppression", "reason": "DO_NOT_CALL"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(analytics, "EVENTS_LOG", events)
    analytics.clear_cache()

    entries = api.get("/api/dashboard/compliance/dnc").json()["entries"]
    assert entries[0]["removable"] is False
    r = api.delete(f"/api/dashboard/compliance/dnc/{entries[0]['id']}")
    assert r.status_code == 400 and "permanent" in r.json()["error"]


# ── Dialling ────────────────────────────────────────────────────────────────

def test_dialling_a_suppressed_number_is_blocked_and_audited(api, monkeypatch):
    api.post("/api/dashboard/compliance/dnc", json={"phone": "+923325026869", "name": "Ayesha"})

    logged = []

    async def fake_block(record):
        logged.append(record)

    async def fail_dial(payload):
        raise AssertionError("a blocked number must never be dialled")

    monkeypatch.setattr("app.utils.logger.log_blocked_call", fake_block)
    monkeypatch.setattr("app.services.twilio_bridge.place_twilio_outbound_call", fail_dial)

    r = api.post("/api/dashboard/calls/dial", json={"lead_id": "L-1"})
    assert r.status_code == 409 and "do-not-call" in r.json()["error"]
    assert logged[0]["rule"] == "Do-Not-Call list"


def test_dial_uses_the_lead_number(api, monkeypatch):
    placed = {}

    async def fake_dial(payload):
        placed.update(payload)
        return {"ok": True, "call_sid": "CA123", "to": payload["to"]}, 200

    monkeypatch.setattr("app.services.twilio_bridge.place_twilio_outbound_call", fake_dial)
    r = api.post("/api/dashboard/calls/dial", json={"lead_id": "L-1"})
    assert r.status_code == 200 and r.json()["call_sid"] == "CA123"
    assert placed["to"] == "+923325026869" and placed["lead_id"] == "L-1"


def test_dial_without_a_number_is_rejected(api):
    assert api.post("/api/dashboard/calls/dial", json={}).status_code == 400


def test_dial_with_unknown_lead_is_a_404(api):
    assert api.post("/api/dashboard/calls/dial", json={"lead_id": "NOPE"}).status_code == 404


# ── Escalation workflow ─────────────────────────────────────────────────────

def test_escalation_can_be_assigned_then_resolved(api, tmp_path, monkeypatch):
    events = tmp_path / "events2.jsonl"
    events.write_text(json.dumps({
        "session_id": "S-ESC", "lead_id": "L-1", "timestamp": "2026-10-07T09:00:00Z",
        "lead_name": "Ayesha Siddiqui", "phone": "+923325026869",
        "type": "callback_request", "reason": "DISPUTE"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(analytics, "EVENTS_LOG", events)
    analytics.clear_cache()

    esc_id = api.get("/api/dashboard/escalations").json()["escalations"][0]["id"]

    api.patch(f"/api/dashboard/escalations/{esc_id}",
              json={"status": "In Progress", "assignedTo": "Ayesha Suleman"})
    row = api.get("/api/dashboard/escalations").json()["escalations"][0]
    assert row["status"] == "In Progress" and row["assignedTo"] == "Ayesha Suleman"

    api.patch(f"/api/dashboard/escalations/{esc_id}",
              json={"status": "Resolved", "resolution": "Called back and agreed a plan"})
    row = api.get("/api/dashboard/escalations").json()["escalations"][0]
    assert row["status"] == "Resolved" and row["resolution"].startswith("Called back")


def test_unknown_escalation_status_is_rejected(api):
    r = api.patch("/api/dashboard/escalations/ESC-1", json={"status": "Sleeping"})
    assert r.status_code == 400


# ── Store durability ────────────────────────────────────────────────────────

def test_writes_survive_a_reload(api, tmp_path):
    api.post("/api/dashboard/customers", json={"name": "Persisted", "phone": "+923009998888"})
    leads_mod._cache.update({"mtime": None, "path": None, "leads": {}})
    names = [c["name"] for c in api.get("/api/dashboard/customers").json()["customers"]]
    assert "Persisted" in names
