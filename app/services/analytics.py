"""Read model for the dashboard.

Reads the JSONL call logs plus the lead store and shapes them into the rows the
dashboard renders. Files are the source of truth and are re-read when they
change, so a call that just ended shows up on the next request.

Nothing here writes. Moving to SQLite later means replacing _read_jsonl() and
the loaders; the aggregation below stays as it is.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.services.leads import list_leads, load_lead, normalize_phone

_BASE_DIR = Path(__file__).resolve().parents[2]
_LOG_DIR = _BASE_DIR / "logs"

SESSIONS_LOG = _LOG_DIR / "call_sessions.jsonl"
TURNS_LOG = _LOG_DIR / "call_transcripts.jsonl"
EVENTS_LOG = _LOG_DIR / "call_events.jsonl"
ATTEMPTS_LOG = _LOG_DIR / "call_attempts.jsonl"

_cache: dict[str, dict] = {}


def _read_jsonl(path: Path) -> list[dict]:
    """Parsed rows from a JSONL file, cached until the file changes."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cached = _cache.get(key)
    if cached and cached["mtime"] == mtime:
        return cached["rows"]

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written line while a call is in flight
    _cache[key] = {"mtime": mtime, "rows": rows}
    return rows


def clear_cache() -> None:
    _cache.clear()


# ── Row accessors ────────────────────────────────────────────────────────────

def sessions() -> list[dict]:
    return [r for r in _read_jsonl(SESSIONS_LOG) if r.get("type") == "session"]


def turns() -> list[dict]:
    return [r for r in _read_jsonl(TURNS_LOG) if r.get("type") == "turn"]


def latencies() -> list[dict]:
    return [r for r in _read_jsonl(TURNS_LOG) if r.get("type") == "latency"]


def events() -> list[dict]:
    return _read_jsonl(EVENTS_LOG)


def attempts() -> list[dict]:
    return _read_jsonl(ATTEMPTS_LOG)


# ── Vocabulary mapping (agent dispositions → dashboard labels) ───────────────

STATUS_BY_DISPOSITION = {
    "ALREADY_PAID": "Success",
    "WRITTEN_DETAILS_REQUESTED": "Success",
    "CALLBACK_SCHEDULED": "Rescheduled",
    "CALLBACK_NO_TIME": "Rescheduled",
    "PREFERENCE_UPDATED": "Rescheduled",
    "ESCALATION_DECLINED": "Escalated",
    "SAFETY_ESCALATION": "Escalated",
    "ATTORNEY_REPRESENTED": "Escalated",
    "BANKRUPTCY": "Escalated",
    "DECEASED": "Escalated",
    "DO_NOT_CALL": "Opt-Out",
    "WRONG_NUMBER": "Wrong Number",
    "NO_RESPONSE": "No Answer",
    "VOICEMAIL_LEFT": "No Answer",
    "IVR_REACHED": "No Answer",
    "TECH_TROUBLE": "Failed",
    # Ending an abusive call is a correct outcome, not a system failure.
    "ENDED_ABUSE": "Abuse",
    "LANGUAGE_BARRIER": "Failed",
    "DECLINED": "Declined",
    "COMPLETED": "Success",
    "CALLER_ENDED": "Declined",
    "MAX_DURATION": "Declined",
    "MAX_TURNS": "Declined",
    # Legacy outcomes from the billing-audit era of the project.
    "SUCCESS_MEETING": "Success",
    "SUCCESS_BAA_OR_EMAIL": "Success",
    "REJECTED_AFTER_3_REFUSALS": "Declined",
    "REJECTED_NOT_INTERESTED": "Declined",
    "REJECTED_ALREADY_HAVE_BILLER_OR_AUDITED": "Declined",
}

INTENT_LABELS = {
    "ALREADY_PAID": "Payment Confirmed",
    "RESCHEDULE": "Reschedule",
    "CALL_BACK_LATER": "Reschedule",
    "TRANSFER_TO_HUMAN": "Escalation Request",
    "DISPUTE": "Escalation Request",
    "HARDSHIP": "Escalation Request",
    "PAYMENT_PLAN": "Escalation Request",
    "WRONG_NUMBER": "Wrong Number",
    "DO_NOT_CALL": "Opt-Out",
    "PARTIAL_OPT_OUT": "Opt-Out",
    "ABUSE": "Abuse",
    "DISTRESS": "Safety Concern",
    "ATTORNEY": "Legal Representation",
    "BANKRUPTCY": "Bankruptcy",
    "DECEASED": "Deceased",
    "GOODBYE": "Caller Ended",
    "TOO_BUSY": "Call Back Later",
    "NOT_INTERESTED": "Not Interested",
    "THIRD_PARTY": "Wrong Person",
    "IDENTITY_DENIAL": "Wrong Person",
    "NOT_DECISION_MAKER": "Wrong Person",
    "LANGUAGE_BARRIER": "Language Barrier",
    "ASK_MORE_INFO": "Asked For Details",
    "TRUST_CONCERN": "Asked For Details",
    "ASK_IF_AI": "Asked For Details",
}

SENTIMENT_BY_MOOD = {
    "positive": "Cooperative",
    "warming": "Cooperative",
    "neutral": "Neutral",
    "skeptical": "Confused",
    "frustrated": "Frustrated",
    "cold": "Hostile",
}

ESCALATION_PRIORITY = {
    "DISTRESS": "High", "ABUSE": "High", "ATTORNEY": "High", "BANKRUPTCY": "High",
    "DISPUTE": "Medium", "HARDSHIP": "Medium", "PAYMENT_PLAN": "Medium",
    "TRANSFER_TO_HUMAN": "Medium", "BAD_EXPERIENCE": "Medium", "PAY_NOW": "Low",
}

ESCALATION_REASONS = {
    "TRANSFER_TO_HUMAN": "Explicit agent request",
    "DISPUTE": "Billing dispute",
    "HARDSHIP": "Billing dispute",
    "PAYMENT_PLAN": "Billing dispute",
    "BAD_EXPERIENCE": "Hostile sentiment detected",
    "ABUSE": "Hostile sentiment detected",
    "DISTRESS": "Hostile sentiment detected",
    "LOOP": "Repeated intent failure",
    "REPEAT_LIMIT": "Low transcription confidence",
}


def status_for(disposition: str | None, total_turns: int = 0) -> str:
    if disposition and disposition in STATUS_BY_DISPOSITION:
        return STATUS_BY_DISPOSITION[disposition]
    return "No Answer" if not total_turns else "Declined"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "0s"
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _lead_of(session: dict, attempt: dict | None = None) -> dict:
    """The lead for a call, falling back to the event log, then the number dialled.

    Calls placed straight from the dialler (or before lead ids existed) have no
    lead_id, so they are matched back to a customer by phone number where possible.
    """
    lead = load_lead(session.get("lead_id") or "")
    if lead:
        return lead

    known = _identity_from_events().get(session.get("session_id") or "", {})
    if known.get("lead_name"):
        return known

    number = (attempt or {}).get("to") or known.get("phone") or ""
    if number:
        match = _lead_by_phone().get(normalize_phone(number))
        if match:
            return match
        return {"lead_name": number, "phone": number}
    return known


def _lead_by_phone() -> dict[str, dict]:
    return {normalize_phone(lead.get("phone", "")): lead for lead in list_leads() if lead.get("phone")}


def _identity_from_events() -> dict[str, dict]:
    identities: dict[str, dict] = {}
    for row in events():
        sid = row.get("session_id")
        if not sid or sid in identities:
            continue
        if row.get("lead_name") or row.get("phone"):
            identities[sid] = {"lead_name": row.get("lead_name"), "phone": row.get("phone") or ""}
    return identities


# ── Calls ────────────────────────────────────────────────────────────────────

def _turns_by_session() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in turns():
        grouped[row.get("session_id", "")].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r.get("turn_index", 0))
    return grouped


def _attempts_by_lead() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in attempts():
        grouped[row.get("lead_id") or ""].append(row)
    return grouped


def _attempt_by_session() -> dict[str, dict]:
    """Twilio's completed-call record, keyed by the session id (= call sid)."""
    out = {}
    for row in attempts():
        if row.get("status") == "completed" and row.get("call_sid"):
            out[row["call_sid"]] = row
    return out


def transcript_of(session_id: str) -> list[dict]:
    """The conversation as [{role: 'Nora'|'Customer', text}] in order."""
    lines: list[dict] = []
    for row in _turns_by_session().get(session_id, []):
        if row.get("user_transcript"):
            lines.append({"role": "Customer", "text": row["user_transcript"]})
        if row.get("llm_response"):
            lines.append({"role": "Nora", "text": row["llm_response"]})
    return lines


def build_call(session: dict, *, with_transcript: bool = False,
               attempts_by_lead: dict | None = None,
               attempt_by_session: dict | None = None) -> dict:
    sid = session.get("session_id", "")
    attempt = (attempt_by_session or {}).get(sid, {})
    lead = _lead_of(session, attempt)
    started = _parse_ts(session.get("start_time"))
    ended = _parse_ts(session.get("end_time"))
    duration_s = attempt.get("duration")
    if duration_s is None and started and ended:
        duration_s = (ended - started).total_seconds()

    moods = session.get("mood_trajectory") or []
    sentiment = SENTIMENT_BY_MOOD.get(moods[-1] if moods else "", "Neutral")
    intent = session.get("last_intent") or "UNCLEAR"
    lead_attempts = (attempts_by_lead or {}).get(session.get("lead_id") or "", [])

    call = {
        "id": sid[:8].upper() if sid else "",
        "sessionId": sid,
        "leadId": session.get("lead_id"),
        "customer": lead.get("lead_name") or lead.get("patient_name") or "Unknown",
        "phone": lead.get("phone") or attempt.get("to") or "",
        "campaign": lead.get("campaign") or "Unassigned",
        "date": started.strftime("%b %d, %Y") if started else "",
        "time": started.strftime("%I:%M %p").lstrip("0") if started else "",
        "startedAt": session.get("start_time"),
        "duration": _fmt_duration(duration_s),
        "durationSeconds": int(duration_s or 0),
        "status": status_for(session.get("final_outcome"), session.get("total_turns", 0)),
        "disposition": session.get("final_outcome"),
        "intent": INTENT_LABELS.get(intent, "Unclear"),
        "rawIntent": intent,
        "sentiment": sentiment,
        "language": lead.get("language_label") or ("English" if (lead.get("language") or "en") == "en" else lead.get("language")),
        # Captured per call once speech confidence is stored (phase 3).
        "confidence": session.get("stt_confidence"),
        "retries": max(0, len(lead_attempts) - 1),
        "turns": session.get("total_turns", 0),
        "identityVerified": bool(session.get("identity_verified")),
        "callbackTime": session.get("callback_time"),
        "escalationReason": session.get("escalation_reason"),
    }
    if with_transcript:
        call["transcript"] = transcript_of(sid)
    return call


def list_calls(limit: int = 200, status: str | None = None, query: str | None = None) -> list[dict]:
    by_lead, by_session = _attempts_by_lead(), _attempt_by_session()
    rows = [build_call(s, attempts_by_lead=by_lead, attempt_by_session=by_session) for s in sessions()]

    # Calls that never connected have no session; surface them from the attempts log.
    session_sids = {s.get("session_id") for s in sessions()}
    for attempt in attempts():
        if attempt.get("status") in {"no-answer", "busy", "failed", "canceled"} \
                and attempt.get("call_sid") not in session_sids:
            lead = load_lead(attempt.get("lead_id") or "") or {}
            started = _parse_ts(attempt.get("timestamp"))
            rows.append({
                "id": (attempt.get("call_sid") or "")[:8].upper(),
                "sessionId": attempt.get("call_sid"),
                "leadId": attempt.get("lead_id"),
                "customer": lead.get("lead_name", "Unknown"),
                "phone": attempt.get("to") or lead.get("phone", ""),
                "campaign": lead.get("campaign") or "Unassigned",
                "date": started.strftime("%b %d, %Y") if started else "",
                "time": started.strftime("%I:%M %p").lstrip("0") if started else "",
                "startedAt": attempt.get("timestamp"),
                "duration": "0s", "durationSeconds": 0,
                "status": "No Answer", "disposition": attempt.get("status", "").upper(),
                "intent": "Unclear", "rawIntent": "UNCLEAR", "sentiment": "Neutral",
                "language": "English", "confidence": None,
                "retries": max(0, len(by_lead.get(attempt.get("lead_id") or "", [])) - 1),
                "turns": 0, "identityVerified": False, "callbackTime": None, "escalationReason": None,
            })

    rows.sort(key=lambda r: r.get("startedAt") or "", reverse=True)
    if status and status != "All":
        rows = [r for r in rows if r["status"] == status]
    if query:
        q = query.lower()
        rows = [r for r in rows if q in r["customer"].lower() or q in r["phone"] or q in r["id"].lower()]
    return rows[:limit]


def get_call(session_id: str) -> dict | None:
    for s in sessions():
        if s.get("session_id") == session_id:
            return build_call(s, with_transcript=True,
                              attempts_by_lead=_attempts_by_lead(),
                              attempt_by_session=_attempt_by_session())
    return None


# ── Overview ────────────────────────────────────────────────────────────────

def _day_key(value: str | None) -> str:
    ts = _parse_ts(value)
    return ts.date().isoformat() if ts else ""


def summary(days: int = 7) -> dict:
    """Metrics over the last `days` days, with today reported separately."""
    calls = list_calls(limit=10_000)
    today = datetime.now(timezone.utc).date().isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days - 1)).date().isoformat()
    todays = [c for c in calls if _day_key(c["startedAt"]) == today]
    scope = [c for c in calls if _day_key(c["startedAt"]) >= cutoff]

    by_status = Counter(c["status"] for c in scope)
    escalated = sum(1 for c in scope if c["status"] == "Escalated" or c["escalationReason"])

    series = []
    for offset in range(days - 1, -1, -1):
        day = (datetime.now(timezone.utc) - timedelta(days=offset)).date()
        day_calls = [c for c in calls if _day_key(c["startedAt"]) == day.isoformat()]
        series.append({
            "day": day.strftime("%a"),
            "date": day.isoformat(),
            "connected": sum(1 for c in day_calls if c["status"] != "No Answer"),
            "noAnswer": sum(1 for c in day_calls if c["status"] == "No Answer"),
        })

    durations = [c["durationSeconds"] for c in scope if c["durationSeconds"]]
    customers = [c for c in list_customers() if c["status"] != "DNC"]

    return {
        "rangeDays": days,
        "totalCalls": len(scope),
        "totalCallsToday": len(todays),
        "totalCallsAllTime": len(calls),
        "resolved": by_status.get("Success", 0),
        "rescheduled": by_status.get("Rescheduled", 0),
        "escalated": escalated,
        "optOuts": by_status.get("Opt-Out", 0),
        "activeCustomers": len(customers),
        "avgHandleSeconds": int(sum(durations) / len(durations)) if durations else 0,
        "avgHandleTime": _fmt_duration(sum(durations) / len(durations) if durations else 0),
        "volume": series,
        "outcomes": {
            "connected": sum(1 for c in scope if c["status"] != "No Answer"),
            "noAnswer": by_status.get("No Answer", 0),
            "abuse": sum(1 for c in scope if c["disposition"] == "ENDED_ABUSE"),
            "escalated": escalated,
        },
        "recentEscalations": list_escalations(limit=5),
    }


# ── Analytics ───────────────────────────────────────────────────────────────

def analytics(days: int = 30) -> dict:
    calls = list_calls(limit=10_000)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    scoped = [c for c in calls if _day_key(c["startedAt"]) >= cutoff] or calls

    trend = []
    for offset in range(days - 1, -1, -1):
        day = (datetime.now(timezone.utc) - timedelta(days=offset)).date()
        day_calls = [c for c in calls if _day_key(c["startedAt"]) == day.isoformat()]
        if not day_calls and offset > 13:
            continue  # don't pad the chart with weeks of zeroes
        trend.append({
            "date": day.strftime("%b %d"),
            "calls": len(day_calls),
            "success": sum(1 for c in day_calls if c["status"] in {"Success", "Rescheduled"}),
            "failed": sum(1 for c in day_calls if c["status"] in {"Failed", "No Answer"}),
        })

    hourly = Counter()
    for c in scoped:
        ts = _parse_ts(c["startedAt"])
        if ts:
            hourly[ts.hour] += 1

    campaigns: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0})
    for c in scoped:
        bucket = campaigns[c["campaign"]]
        bucket["total"] += 1
        bucket["success"] += 1 if c["status"] in {"Success", "Rescheduled"} else 0

    weeks: dict[str, list] = defaultdict(list)
    for c in scoped:
        ts = _parse_ts(c["startedAt"])
        if ts:
            weeks[ts.strftime("%Y-W%V")].append(c)

    latency_rows = latencies()
    stage_avg: dict[str, int] = {}
    if latency_rows:
        buckets: dict[str, list[int]] = defaultdict(list)
        for row in latency_rows:
            for stage, ms in (row.get("latency_ms") or {}).items():
                buckets[stage].append(ms)
        stage_avg = {k: int(sum(v) / len(v)) for k, v in sorted(buckets.items())}

    return {
        "callVolumeTrend": trend,
        "outcomes": [{"name": name, "value": count}
                     for name, count in Counter(c["status"] for c in scoped).most_common()],
        "intents": [{"intent": name, "count": count}
                    for name, count in Counter(c["intent"] for c in scoped).most_common()],
        "sentiment": [{"sentiment": name, "value": count}
                      for name, count in Counter(c["sentiment"] for c in scoped).most_common()],
        "campaigns": [{"campaign": name, "total": v["total"], "success": v["success"],
                       "rate": round(v["success"] / v["total"] * 100, 1) if v["total"] else 0.0}
                      for name, v in sorted(campaigns.items())],
        "hourly": [{"hour": f"{h:02d}:00", "calls": hourly.get(h, 0)} for h in range(24)],
        "languages": [{"name": name, "value": count}
                      for name, count in Counter(c["language"] for c in scoped).most_common()],
        "successTrend": [
            {"week": week.split("-")[-1],
             "rate": round(sum(1 for c in rows if c["status"] in {"Success", "Rescheduled"}) / len(rows) * 100, 1)}
            for week, rows in sorted(weeks.items())
        ],
        "latencyMs": stage_avg,
        "totals": {
            "calls": len(scoped),
            "successRate": round(
                sum(1 for c in scoped if c["status"] in {"Success", "Rescheduled"}) / len(scoped) * 100, 1
            ) if scoped else 0.0,
            "avgTurns": round(sum(c["turns"] for c in scoped) / len(scoped), 1) if scoped else 0.0,
        },
    }


# ── Customers ───────────────────────────────────────────────────────────────

def _amount_value(raw: Any) -> float:
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


def list_customers() -> list[dict]:
    dnc_numbers = {entry["phone"] for entry in list_dnc()}
    by_lead_sessions: dict[str, list[dict]] = defaultdict(list)
    for s in sessions():
        by_lead_sessions[s.get("lead_id") or ""].append(s)
    attempts_by_lead = _attempts_by_lead()

    rows = []
    for lead in list_leads():
        lead_id = lead["lead_id"]
        lead_sessions = sorted(by_lead_sessions.get(lead_id, []), key=lambda s: s.get("start_time") or "")
        last = _parse_ts(lead_sessions[-1]["start_time"]) if lead_sessions else None
        due = lead.get("due_date") or ""

        if lead.get("phone") in dnc_numbers:
            status = "DNC"
        elif lead.get("amount_due"):
            status = "Overdue"
        elif lead_sessions:
            status = "Active"
        else:
            status = "Inactive"

        rows.append({
            "id": lead_id,
            "name": lead.get("lead_name") or lead.get("patient_name") or "Unknown",
            "email": lead.get("email", ""),
            "phone": lead.get("phone", ""),
            "campaign": lead.get("campaign") or "Unassigned",
            "amountDue": _amount_value(lead.get("amount_due")),
            "amountDueLabel": lead.get("amount_due", ""),
            "dueDate": due,
            "status": status,
            "lastContacted": last.strftime("%b %d, %Y") if last else "Never",
            "attempts": len(attempts_by_lead.get(lead_id, [])) or len(lead_sessions),
        })
    return rows


# ── Compliance ──────────────────────────────────────────────────────────────

DNC_SOURCE_BY_REASON = {
    "DO_NOT_CALL": "Customer opt-out",
    "WRONG_NUMBER": "Wrong number",
    "ATTORNEY": "Manual entry",
    "BANKRUPTCY": "Manual entry",
    "DECEASED": "Manual entry",
}


def list_dnc() -> list[dict]:
    """Numbers the agent must not call again, newest first."""
    entries: dict[str, dict] = {}
    for row in events():
        if row.get("type") != "suppression":
            continue
        phone = row.get("phone") or ""
        # Keyed by number when known, otherwise by customer, so an opt-out is
        # never dropped just because the record had no phone on file.
        key = phone or row.get("lead_id") or row.get("session_id") or ""
        if not key:
            continue
        ts = _parse_ts(row.get("timestamp"))
        entries[key] = {
            "id": f"DNC-{len(entries) + 1:04d}",
            "phone": phone or "Not on file",
            "name": row.get("lead_name") or "Unknown",
            "leadId": row.get("lead_id"),
            "source": DNC_SOURCE_BY_REASON.get(row.get("reason", ""), "Manual entry"),
            "addedOn": ts.strftime("%b %d, %Y") if ts else "",
            "addedAt": row.get("timestamp"),
            "addedBy": "NORA (automated)",
            "note": f"Recorded on call {row.get('session_id', '')[:8]} ({row.get('reason', '')})",
            "removable": False,
        }
    from app.services.stores import read_dnc
    for manual in read_dnc():
        phone = manual.get("phone") or ""
        ts = _parse_ts(manual.get("addedAt"))
        entries[phone or manual.get("id", "")] = {
            "id": manual.get("id", ""),
            "phone": phone or "Not on file",
            "name": manual.get("name", "Unknown"),
            "leadId": None,
            "source": manual.get("source", "Manual entry"),
            "addedOn": ts.strftime("%b %d, %Y") if ts else "",
            "addedAt": manual.get("addedAt"),
            "addedBy": manual.get("addedBy", "Dashboard user"),
            "note": manual.get("note", ""),
            "removable": True,
        }
    return sorted(entries.values(), key=lambda e: e.get("addedAt") or "", reverse=True)


def list_audit() -> list[dict]:
    """Calls blocked before dialling. Populated once pre-dial checks land."""
    rows = []
    for row in events():
        if row.get("type") != "blocked_call":
            continue
        ts = _parse_ts(row.get("timestamp"))
        rows.append({
            "id": row.get("id") or f"AUD-{len(rows) + 1:04d}",
            "time": ts.strftime("%I:%M %p").lstrip("0") if ts else "",
            "date": ts.strftime("%b %d, %Y") if ts else "",
            "phone": row.get("phone", ""),
            "rule": row.get("rule", ""),
            "campaign": row.get("campaign", "Unassigned"),
            "action": row.get("action", "Call blocked"),
        })
    return rows


# ── Escalations ─────────────────────────────────────────────────────────────

def list_escalations(limit: int = 100) -> list[dict]:
    sessions_by_id = {s.get("session_id"): s for s in sessions()}
    rows = []
    for row in events():
        if row.get("type") not in {"escalation", "callback_request"}:
            continue
        sid = row.get("session_id", "")
        session = sessions_by_id.get(sid, {})
        lead = load_lead(row.get("lead_id") or "") or {}
        raised = _parse_ts(row.get("timestamp"))
        reason_key = row.get("reason", "")
        waiting = int((datetime.now(timezone.utc) - raised).total_seconds() / 60) if raised else 0
        rows.append({
            "id": f"ESC-{sid[:6].upper()}",
            "callId": sid,
            "customer": row.get("lead_name") or lead.get("lead_name") or "Unknown",
            "phone": row.get("phone") or lead.get("phone", ""),
            "campaign": lead.get("campaign") or "Unassigned",
            "reason": ESCALATION_REASONS.get(reason_key, "Explicit agent request"),
            "rawReason": reason_key,
            "priority": ESCALATION_PRIORITY.get(reason_key, "Medium"),
            "status": "Open",
            "raisedAt": row.get("timestamp"),
            "raisedLabel": raised.strftime("%b %d, %I:%M %p") if raised else "",
            "waitingMins": max(0, waiting),
            "slaMins": 60,
            "assignedTo": None,
            "requestedTime": row.get("requested_time"),
            "language": "English",
            "sentiment": SENTIMENT_BY_MOOD.get((session.get("mood_trajectory") or ["neutral"])[-1], "Neutral"),
            "confidence": session.get("stt_confidence"),
        })
    from app.services.stores import read_escalation_state
    state = read_escalation_state()
    for row in rows:
        saved = state.get(row["id"])
        if saved:
            row["status"] = saved.get("status", row["status"])
            row["assignedTo"] = saved.get("assignedTo", row["assignedTo"])
            if saved.get("resolution"):
                row["resolution"] = saved["resolution"]
            row["updatedAt"] = saved.get("updatedAt")

    rows.sort(key=lambda r: r.get("raisedAt") or "", reverse=True)
    return rows[:limit]
