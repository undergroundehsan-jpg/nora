"""API behind the analytics dashboard.

Reads aggregate the agent's call logs and lead store; writes manage customers,
the do-not-call list, escalation workflow state, and outbound dialling.
Every response is shaped the way the dashboard renders it.
"""
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

from app.services import analytics, stores
from app.services.leads import LeadError, lead_for_call, list_leads, phone_for_lead

dashboard_router = APIRouter()


@dashboard_router.get("/summary")
async def get_summary(days: int = Query(7, ge=1, le=90)) -> dict:
    """Overview page: metric cards, call-volume chart, outcome split."""
    return analytics.summary(days=days)


@dashboard_router.get("/calls")
async def get_calls(
    limit: int = Query(200, ge=1, le=2000),
    status: str | None = None,
    q: str | None = None,
) -> dict:
    rows = analytics.list_calls(limit=limit, status=status, query=q)
    return {"calls": rows, "count": len(rows)}


@dashboard_router.get("/calls/{session_id}")
async def get_call(session_id: str) -> dict:
    call = analytics.get_call(session_id)
    return {"call": call} if call else {"call": None, "error": "Call not found"}


@dashboard_router.get("/analytics")
async def get_analytics(days: int = Query(30, ge=1, le=365)) -> dict:
    return analytics.analytics(days=days)


@dashboard_router.get("/customers")
async def get_customers() -> dict:
    rows = analytics.list_customers()
    return {"customers": rows, "count": len(rows)}


@dashboard_router.get("/escalations")
async def get_escalations(limit: int = Query(100, ge=1, le=500)) -> dict:
    rows = analytics.list_escalations(limit=limit)
    return {"escalations": rows, "count": len(rows)}


@dashboard_router.get("/compliance/dnc")
async def get_dnc() -> dict:
    rows = analytics.list_dnc()
    return {"entries": rows, "count": len(rows)}


@dashboard_router.get("/compliance/audit")
async def get_audit() -> dict:
    rows = analytics.list_audit()
    return {"events": rows, "count": len(rows)}


@dashboard_router.get("/campaigns")
async def get_campaigns() -> dict:
    """Campaigns are derived from the campaign field on each lead for now."""
    stats = {row["campaign"]: row for row in analytics.analytics(days=365)["campaigns"]}
    campaigns: dict[str, dict] = {}
    for lead in list_leads():
        name = lead.get("campaign") or "Unassigned"
        bucket = campaigns.setdefault(name, {
            "id": name.lower().replace(" ", "-"),
            "name": name,
            "type": lead.get("call_type", "REMINDER").title(),
            "status": "Active",
            "total": 0, "called": 0, "confirmed": 0, "escalated": 0, "failed": 0,
            "language": lead.get("language", "en"),
        })
        bucket["total"] += 1
    for name, bucket in campaigns.items():
        stat = stats.get(name, {})
        bucket["called"] = stat.get("total", 0)
        bucket["confirmed"] = stat.get("success", 0)
        bucket["rate"] = stat.get("rate", 0.0)
    return {"campaigns": list(campaigns.values()), "count": len(campaigns)}


# ── Writes ──────────────────────────────────────────────────────────────────

def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


@dashboard_router.post("/customers")
async def create_customer(payload: dict = Body(...)):
    """Add one customer to the lead store."""
    try:
        customer = stores.add_lead(payload)
    except LeadError as e:
        return _error(str(e))
    return {"ok": True, "customer": customer}


@dashboard_router.patch("/customers/{lead_id}")
async def edit_customer(lead_id: str, payload: dict = Body(...)):
    try:
        customer = stores.update_lead(lead_id, payload)
    except LeadError as e:
        return _error(str(e), 404 if "Unknown" in str(e) else 400)
    return {"ok": True, "customer": customer}


@dashboard_router.delete("/customers/{lead_id}")
async def remove_customer(lead_id: str):
    try:
        stores.delete_lead(lead_id)
    except LeadError as e:
        return _error(str(e), 404)
    return {"ok": True}


@dashboard_router.post("/customers/import")
async def import_customers(payload: dict = Body(...)):
    """Bulk import from the dashboard's CSV upload."""
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return _error("Expected 'rows' to be a list of customers")
    return {"ok": True, **stores.import_leads(rows)}


@dashboard_router.post("/compliance/dnc")
async def add_dnc_entry(payload: dict = Body(...)):
    try:
        entry = stores.add_dnc(payload)
    except LeadError as e:
        return _error(str(e))
    return {"ok": True, "entry": entry}


@dashboard_router.delete("/compliance/dnc/{entry_id}")
async def remove_dnc_entry(entry_id: str):
    try:
        stores.remove_dnc(entry_id)
    except LeadError as e:
        return _error(str(e))
    return {"ok": True}


@dashboard_router.patch("/escalations/{escalation_id}")
async def update_escalation(escalation_id: str, payload: dict = Body(...)):
    """Assign an escalation to someone, or close it with a resolution note."""
    status = payload.get("status")
    if status and status not in {"Open", "In Progress", "Resolved"}:
        return _error(f"Unknown status '{status}'")
    return {"ok": True, "escalation": stores.set_escalation_state(escalation_id, payload)}


@dashboard_router.post("/calls/dial")
async def dial(payload: dict = Body(...)):
    """Place an outbound call for a customer.

    Refuses numbers on the do-not-call list and records the refusal in the
    compliance audit log, so a blocked call is visible rather than silent.
    """
    from app.services.twilio_bridge import place_twilio_outbound_call
    from app.utils.logger import log_blocked_call

    lead_id = str(payload.get("lead_id") or "").strip()
    to_number = str(payload.get("to") or "").strip()

    lead = {}
    if lead_id:
        try:
            lead = lead_for_call(lead_id)
        except LeadError as e:
            return _error(str(e), 404)
        to_number = to_number or phone_for_lead(lead)

    if not to_number:
        return _error("No number to call: give a lead_id with a phone number, or a 'to' number")

    # Dialled by number alone: match it back to a customer so the call is
    # attributed on the dashboard instead of showing as "Unknown".
    if not lead_id:
        from app.services.leads import normalize_phone
        target = normalize_phone(to_number)
        for candidate in list_leads():
            if normalize_phone(candidate.get("phone", "")) == target:
                lead, lead_id = candidate, candidate["lead_id"]
                break

    if stores.is_suppressed(to_number):
        await log_blocked_call({
            "phone": to_number,
            "lead_id": lead_id or None,
            "campaign": lead.get("campaign", "Unassigned"),
            "rule": "Do-Not-Call list",
            "action": "Call blocked before dialling",
        })
        return _error(f"{to_number} is on the do-not-call list, so the call was blocked", 409)

    result, status_code = await place_twilio_outbound_call({"lead_id": lead_id, "to": to_number})
    return JSONResponse(result, status_code=status_code)
