"""Build the call history that ships with the deployed backend.

The logs in `logs/` go back to March, from before calls carried a customer
record, so their rows have no `lead_id`. The dashboard resolves the customer
from that field, so those calls show up as "Unknown / Unassigned". This writes
a processed copy into `seed/` with the customer stamped on, leaving the real
logs untouched.

    python scripts/build_seed_logs.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.leads import list_leads  # noqa: E402

LOGS = ROOT / "logs"
SEED = ROOT / "seed"

# Rows that name a customer in the dashboard.
STAMPED = {"call_sessions.jsonl", "call_attempts.jsonl", "call_transcripts.jsonl"}


def main() -> None:
    leads = list_leads()
    if not leads:
        raise SystemExit("data/leads.json has no customers; nothing to stamp")
    lead = leads[0]
    lead_id, name, phone = lead["lead_id"], lead.get("lead_name", ""), lead.get("phone", "")
    # Rows may also carry a lead_id from a customer list that no longer exists
    # (the sample customer, or a dataset since removed). Those resolve to
    # nothing, so they are remapped too.
    known_ids = {l["lead_id"] for l in leads}

    if SEED.exists():
        shutil.rmtree(SEED)
    SEED.mkdir()

    for source in sorted(LOGS.glob("*.jsonl")):
        rows, stamped = [], 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written line from an interrupted run
            if source.name in STAMPED and row.get("lead_id") not in known_ids:
                row["lead_id"] = lead_id
                stamped += 1
            if source.name == "call_attempts.jsonl" and not row.get("to"):
                row["to"] = phone
            rows.append(row)

        target = SEED / source.name
        target.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        print(f"  {source.name:28} {len(rows):4} rows, {stamped:4} stamped as {lead_id}")

    print(f"\nSeed written to {SEED} — every historical call now shows as {name}.")


if __name__ == "__main__":
    main()
