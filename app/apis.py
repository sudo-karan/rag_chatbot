"""Implementations of the five mocked API intents.

All five functions are still `str -> str` so the chat handler and intent
router are unchanged from earlier PRs. The data behind the responses is
now real:

- search / cdo_details / dataset_cdo_link are backed by app/datasets.py
  which loads a CSV (or JSON) directory of {title, url, contributor_name,
  contributor_email, ...} records.
- submit_portal_feedback and contact_cdo_or_dataset_feedback append a
  JSONL record to FEEDBACK_LOG_FILE and return a ticket id.

When the dataset directory is not configured, the read functions return
a polite "not configured" message; the write functions still work.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import FEEDBACK_LOG_FILE
from app.datasets import (
    DatasetRecord,
    get_by_contributor,
    get_by_url,
    is_configured,
    search,
)


_NOT_CONFIGURED = (
    "The dataset directory is not configured yet. Please ask the portal "
    "administrator to add datasets.csv with title, url, contributor_name, "
    "and contributor_email columns."
)


def _format_record_short(r: DatasetRecord) -> list[str]:
    lines = [f"   URL: {r.url}",
             f"   Contributor: {r.contributor_name} ({r.contributor_email})"]
    if r.ministry:
        lines.append(f"   Ministry: {r.ministry}")
    if r.last_updated:
        lines.append(f"   Last updated: {r.last_updated}")
    return lines


def search_datasets(keyword: str) -> str:
    keyword = keyword.strip()
    if not keyword:
        return "Please provide a search term to find datasets."
    if not is_configured():
        return _NOT_CONFIGURED

    results = search(keyword, n=5)
    if not results:
        return (
            f"I couldn't find any datasets matching '{keyword}'. "
            "Try a different keyword, or check data.gov.in directly."
        )

    lines = [f"Top results for '{keyword}':", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.extend(_format_record_short(r))
        lines.append("")
    return "\n".join(lines).rstrip()


def get_cdo_details(query: str) -> str:
    query = query.strip()
    if not query:
        return "Please specify a name, ministry, department, or email to look up."
    if not is_configured():
        return _NOT_CONFIGURED

    matches = get_by_contributor(query, n=20)
    if not matches:
        return (
            f"I couldn't find a Chief Data Officer or contributor matching "
            f"'{query}'. Try an exact name, ministry, or email."
        )

    seen: dict[str, list[DatasetRecord]] = {}
    for r in matches:
        seen.setdefault(r.contributor_email or r.contributor_name, []).append(r)

    lines = [f"Contributor / CDO details for '{query}':", ""]
    for key, recs in list(seen.items())[:5]:
        head = recs[0]
        lines.append(f"Name: {head.contributor_name}")
        lines.append(f"Email: {head.contributor_email}")
        if head.ministry:
            lines.append(f"Ministry: {head.ministry}")
        lines.append(f"Datasets contributed: {len(recs)}")
        for r in recs[:3]:
            lines.append(f"  - {r.title}")
        if len(recs) > 3:
            lines.append(f"  ... and {len(recs) - 3} more")
        lines.append("")
    return "\n".join(lines).rstrip()


def get_dataset_cdo(dataset_url: str) -> str:
    dataset_url = dataset_url.strip()
    if not dataset_url:
        return "Please provide a data.gov.in dataset URL."
    if not is_configured():
        return _NOT_CONFIGURED

    record = get_by_url(dataset_url)
    if not record:
        return (
            f"I couldn't find a dataset matching '{dataset_url}' in the "
            "directory. Please verify the URL or browse the catalog on data.gov.in."
        )

    lines = [
        f"Dataset: {record.title}",
        f"URL: {record.url}",
        "",
        f"Contributed by: {record.contributor_name}",
        f"Email: {record.contributor_email}",
    ]
    if record.ministry:
        lines.append(f"Ministry: {record.ministry}")
    if record.last_updated:
        lines.append(f"Last updated: {record.last_updated}")
    return "\n".join(lines)


def _new_ticket_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"


def _log_feedback(payload: dict) -> str:
    """Append a JSONL record to FEEDBACK_LOG_FILE. Returns the ticket id."""
    ticket_id = payload["ticket_id"]
    log_path = Path(FEEDBACK_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"WARNING: Could not write feedback log to {log_path}: {e}")
    return ticket_id


def submit_portal_feedback(message: str = "") -> str:
    ticket_id = _new_ticket_id("PFB")
    _log_feedback({
        "ticket_id": ticket_id,
        "kind": "portal_feedback",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "message": message.strip(),
    })
    return (
        f"Your feedback about data.gov.in has been recorded (ticket {ticket_id}). "
        "Our team will review it within 3-5 working days. Thank you for helping "
        "us improve the portal."
    )


def contact_cdo_or_dataset_feedback(dataset_ref: str = "") -> str:
    dataset_ref = dataset_ref.strip()
    record: DatasetRecord | None = None
    if dataset_ref:
        record = get_by_url(dataset_ref)
        if record is None:
            candidates = get_by_contributor(dataset_ref, n=1)
            record = candidates[0] if candidates else None

    ticket_id = _new_ticket_id("DFB" if dataset_ref else "CDO")
    _log_feedback({
        "ticket_id": ticket_id,
        "kind": "dataset_feedback" if dataset_ref else "cdo_contact",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_ref": dataset_ref,
        "routed_to": (
            {"name": record.contributor_name, "email": record.contributor_email}
            if record else None
        ),
    })

    if record:
        return (
            f"Your feedback regarding '{record.title}' has been forwarded to "
            f"{record.contributor_name} ({record.contributor_email}). "
            f"Ticket: {ticket_id}. You will receive a response within 5 working days."
        )
    if dataset_ref:
        return (
            f"I couldn't find a dataset matching '{dataset_ref}' in the directory, "
            f"but your message has been recorded (ticket {ticket_id}) and will be "
            "routed to the relevant Chief Data Officer. Response in 5 working days."
        )
    return (
        f"Your message has been recorded (ticket {ticket_id}) and will be routed "
        "to the relevant Chief Data Officer. Response in 5 working days."
    )
