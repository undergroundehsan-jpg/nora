"""Writable JSON stores behind the dashboard.

Each store is a small JSON file written atomically (temp file + replace) under a
lock, so a half-written file can never be read by a call that is in flight.

Everything here is deliberately boring: swapping to SQLite later means
reimplementing these functions, not changing their callers.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.leads import LeadError, _leads_path, build_lead_context, normalize_phone

_BASE_DIR = Path(__file__).resolve().parents[2]
_DATA_DIR = _BASE_DIR / "data"

DNC_PATH = _DATA_DIR / "dnc.json"
ESCALATION_STATE_PATH = _DATA_DIR / "escalation_state.json"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    """Write atomically so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ── Leads ───────────────────────────────────────────────────────────────────

def _read_leads() -> list[dict]:
    data = read_json(_leads_path(), [])
    return list(data.values()) if isinstance(data, dict) else list(data)


def _save_leads(records: list[dict]) -> None:
    write_json(_leads_path(), records)
    from app.services import leads as leads_mod
    leads_mod._cache.update({"mtime": None, "path": None, "leads": {}})


def _next_lead_id(records: list[dict]) -> str:
    numbers = []
    for record in records:
        raw = str(record.get("lead_id", ""))
        if raw.startswith("L-") and raw[2:].isdigit():
            numbers.append(int(raw[2:]))
    return f"L-{(max(numbers) + 1) if numbers else 1001}"


def add_lead(payload: dict) -> dict:
    """Create a customer. Name and a usable phone number are required."""
    name = str(payload.get("name") or payload.get("lead_name") or "").strip()
    phone = normalize_phone(payload.get("phone", ""))
    if not name:
        raise LeadError("A customer name is required")
    if not phone:
        raise LeadError("A phone number in international format is required (e.g. +923001234567)")

    with _lock:
        records = _read_leads()
        if any(normalize_phone(r.get("phone", "")) == phone for r in records):
            raise LeadError(f"A customer with phone {phone} already exists")
        record = {
            "lead_id": payload.get("lead_id") or _next_lead_id(records),
            "lead_name": name,
            "phone": phone,
            "practice_name": payload.get("practice_name") or payload.get("organisation") or "the practice",
            "email": payload.get("email", ""),
            "campaign": payload.get("campaign") or "Unassigned",
            "amount_due": payload.get("amount_due") or payload.get("amountDue") or "",
            "due_date": payload.get("due_date") or payload.get("dueDate") or "",
            "call_type": "REMINDER",
            "created_at": _now(),
        }
        records.append({k: v for k, v in record.items() if v not in (None, "")})
        _save_leads(records)
    return build_lead_context(record)


def update_lead(lead_id: str, patch: dict) -> dict:
    field_names = {
        "name": "lead_name", "lead_name": "lead_name", "email": "email", "phone": "phone",
        "campaign": "campaign", "amountDue": "amount_due", "amount_due": "amount_due",
        "dueDate": "due_date", "due_date": "due_date", "practice_name": "practice_name",
    }
    with _lock:
        records = _read_leads()
        for record in records:
            if str(record.get("lead_id")) != str(lead_id):
                continue
            for key, value in patch.items():
                field = field_names.get(key)
                if not field:
                    continue
                if field == "phone":
                    value = normalize_phone(value)
                    if not value:
                        raise LeadError("A phone number in international format is required")
                record[field] = value
            record["updated_at"] = _now()
            _save_leads(records)
            return build_lead_context(record)
    raise LeadError(f"Unknown lead_id '{lead_id}'")


def delete_lead(lead_id: str) -> None:
    with _lock:
        records = _read_leads()
        remaining = [r for r in records if str(r.get("lead_id")) != str(lead_id)]
        if len(remaining) == len(records):
            raise LeadError(f"Unknown lead_id '{lead_id}'")
        _save_leads(remaining)


def import_leads(rows: list[dict]) -> dict:
    """Bulk import from the dashboard's CSV upload.

    Rows that fail validation are reported back with their index rather than
    silently dropped, and nothing is written unless at least one row is valid.
    """
    created, errors = [], []
    for index, row in enumerate(rows):
        try:
            created.append(add_lead(row))
        except LeadError as e:
            errors.append({"row": index + 1, "error": str(e),
                           "name": row.get("name") or row.get("lead_name", "")})
    return {"imported": len(created), "failed": len(errors), "customers": created, "errors": errors}


# ── Do-not-call list ────────────────────────────────────────────────────────

def read_dnc() -> list[dict]:
    """Manually managed entries (the automatic ones come from call events)."""
    return read_json(DNC_PATH, [])


def add_dnc(payload: dict) -> dict:
    phone = normalize_phone(payload.get("phone", ""))
    if not phone:
        raise LeadError("A phone number in international format is required")
    with _lock:
        entries = read_dnc()
        if any(e.get("phone") == phone for e in entries):
            raise LeadError(f"{phone} is already on the do-not-call list")
        entry = {
            "id": f"DNC-M-{len(entries) + 1:04d}",
            "phone": phone,
            "name": payload.get("name") or "Unknown",
            "source": payload.get("source") or "Manual entry",
            "addedBy": payload.get("addedBy") or "Dashboard user",
            "note": payload.get("note") or "",
            "addedAt": _now(),
        }
        entries.append(entry)
        write_json(DNC_PATH, entries)
    return entry


def remove_dnc(entry_id: str) -> None:
    """Remove a manually added entry. Automatic opt-outs cannot be removed here."""
    with _lock:
        entries = read_dnc()
        remaining = [e for e in entries if e.get("id") != entry_id and e.get("phone") != entry_id]
        if len(remaining) == len(entries):
            raise LeadError(
                "Only manually added entries can be removed. Opt-outs recorded on a call are permanent."
            )
        write_json(DNC_PATH, remaining)


def is_suppressed(phone: str) -> bool:
    """True when this number must not be called (manual list or a recorded opt-out)."""
    from app.services.analytics import list_dnc
    target = normalize_phone(phone)
    if not target:
        return False
    numbers = {normalize_phone(e.get("phone", "")) for e in list_dnc()}
    numbers |= {normalize_phone(e.get("phone", "")) for e in read_dnc()}
    return target in numbers


# ── Escalation workflow state ───────────────────────────────────────────────

def read_escalation_state() -> dict:
    return read_json(ESCALATION_STATE_PATH, {})


def set_escalation_state(escalation_id: str, patch: dict) -> dict:
    allowed = {"status", "assignedTo", "resolution"}
    with _lock:
        state = read_escalation_state()
        entry = state.get(escalation_id, {})
        entry.update({k: v for k, v in patch.items() if k in allowed})
        entry["updatedAt"] = _now()
        state[escalation_id] = entry
        write_json(ESCALATION_STATE_PATH, state)
    return entry
