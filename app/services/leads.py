"""Customer (lead) records for outbound reminder calls.

The agent is given one lead per call. A lead carries who to greet, what the
reminder is about, and the identifiers used to confirm who is speaking.

Storage is a JSON file (LEADS_FILE, default data/leads.json) shaped as either
a list of records or an object keyed by lead_id. It is re-read whenever the
file changes, so edits apply without restarting the server.

Swapping to SQLite or Supabase later only means replacing _read_store(); the
rest of the application depends on load_lead()/list_leads() alone.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from app.session import SAMPLE_LEAD_CONTEXT

_BASE_DIR = Path(__file__).resolve().parents[2]


def _leads_path() -> Path:
    raw = os.getenv("LEADS_FILE", "data/leads.json").strip() or "data/leads.json"
    path = Path(raw)
    return path if path.is_absolute() else _BASE_DIR / path


def allow_sample_lead() -> bool:
    """Dev convenience: fall back to the sample customer when no lead is given."""
    return os.getenv("ALLOW_SAMPLE_LEAD", "true").strip().lower() not in {"0", "false", "no"}


# Fields copied straight through to the prompt / greeting.
_TEXT_FIELDS = (
    "lead_name", "practice_name", "patient_name", "patient_dob", "member_id",
    "provider_npi", "tax_id", "email", "city", "specialty", "designation",
    "service", "amount_due", "due_date", "appointment_time", "timezone", "language",
    "call_type", "campaign",
)

_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")


class LeadError(ValueError):
    """Raised when a lead exists but cannot be used for a call."""


_cache: dict = {"mtime": None, "path": None, "leads": {}}


def _read_store() -> dict:
    """Return {lead_id: raw record}, re-reading the file when it changes."""
    path = _leads_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _cache["mtime"] == mtime and _cache["path"] == str(path):
        return _cache["leads"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[LEADS] Could not read {path}: {e}")
        return {}

    records = data.values() if isinstance(data, dict) else data
    leads = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        lead_id = str(rec.get("lead_id") or "").strip()
        if lead_id:
            leads[lead_id] = rec

    _cache.update({"mtime": mtime, "path": str(path), "leads": leads})
    print(f"[LEADS] Loaded {len(leads)} lead(s) from {path}")
    return leads


def normalize_phone(raw: str) -> str:
    """Return the number in E.164 form, or '' when it cannot be trusted.

    Accepts "+92332…", "0092 332…" and "92-332-…". A bare national number
    ("0332…") is rejected, because guessing its country could dial a stranger.
    """
    cleaned = re.sub(r"[^\d+]", "", str(raw or ""))
    if cleaned.startswith("00"):  # international prefix
        cleaned = "+" + cleaned[2:]
    elif cleaned and not cleaned.startswith("+"):
        cleaned = "+" + cleaned
    return cleaned if _E164_RE.match(cleaned) else ""


def build_lead_context(record: dict) -> dict:
    """Turn a stored record into the lead_context the agent uses.

    Only fields that are present are included, so the prompt never claims to
    know an amount or due date that isn't on file.
    """
    context = {"lead_id": str(record.get("lead_id") or "").strip()}
    for field in _TEXT_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            context[field] = str(value).strip()

    phone = normalize_phone(record.get("phone", ""))
    if phone:
        context["phone"] = phone

    context.setdefault("call_type", "REMINDER")
    context.setdefault("practice_name", "the practice")
    return context


def load_lead(lead_id: str) -> Optional[dict]:
    """Return the lead_context for `lead_id`, or None when it is unknown."""
    if not lead_id:
        return None
    record = _read_store().get(str(lead_id).strip())
    return build_lead_context(record) if record else None


def list_leads() -> list[dict]:
    """Every lead as a context dict (used by tooling and the dashboard)."""
    return [build_lead_context(r) for r in _read_store().values()]


def lead_for_call(lead_id: str = "", fallback_to_sample: Optional[bool] = None) -> dict:
    """Resolve the lead for a call.

    Raises LeadError when the id is unknown, or when no id was given and the
    sample customer is not allowed — better to fail before dialling than to
    call a real person with placeholder details.
    """
    allow_sample = allow_sample_lead() if fallback_to_sample is None else fallback_to_sample

    if lead_id:
        lead = load_lead(lead_id)
        if lead is None:
            raise LeadError(f"Unknown lead_id '{lead_id}' in {_leads_path().name}")
        return lead

    if not allow_sample:
        raise LeadError("No lead_id supplied and ALLOW_SAMPLE_LEAD is disabled")

    sample = dict(SAMPLE_LEAD_CONTEXT)
    sample.setdefault("lead_id", "SAMPLE")
    return sample


def phone_for_lead(lead: dict) -> str:
    """The number to dial for this lead ('' when it has none)."""
    return normalize_phone(lead.get("phone", ""))
