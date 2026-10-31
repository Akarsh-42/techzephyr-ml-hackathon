"""Step 0 — Continuous IST audit logger.

Dual-logger that appends every coded feature and every pipeline run
(success AND failure) to `done_so_far.md` and `what_we_did.md` with
Asia/Kolkata (IST, UTC+5:30) timestamps. This is the copy-paste source
material for the 15-point README rubric.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DONE_SO_FAR_PATH = PROJECT_ROOT / "done_so_far.md"
WHAT_WE_DID_PATH = PROJECT_ROOT / "what_we_did.md"

IST = ZoneInfo("Asia/Kolkata")  # UTC+5:30


def _ist_now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _append_markdown(path: Path, block: str) -> None:
    """Create the audit file with a header on first write, then append."""
    if not path.exists():
        header = (
            f"# Sketch2Shape AI — {'Continuous Audit Log' if path == DONE_SO_FAR_PATH else 'Role Importance, Jobs & Architecture'}"
            f" ({path.name})\n\n"
        )
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(block)


def log_feature_progress(
    feature_name: str,
    role_owner: str,
    role_importance_and_job: str,
    files_modified: list[str],
    technical_explanation: str,
    duration_or_latency: str = "N/A",
) -> None:
    """Appends a timestamped feature completion entry (in IST) to done_so_far.md and what_we_did.md."""
    block = (
        f"### [{_ist_now()}] — Feature: {feature_name}\n"
        f"- Role Owner: {role_owner}\n"
        f"- Role Importance & Job: {role_importance_and_job}\n"
        f"- Files Touched: {', '.join(files_modified)}\n"
        f"- Time Taken / Latency: {duration_or_latency}\n"
        f"- Technical Explanation: {technical_explanation}\n"
        f"---\n"
    )
    _append_markdown(DONE_SO_FAR_PATH, block)
    _append_markdown(WHAT_WE_DID_PATH, block)


def log_pipeline_run(
    run_id: str,
    status: str,  # "SUCCESS", "RETRY_SUCCESS", or "FAILURE"
    stage_timings: dict,
    mesh_report: dict,
    notes: str = "",
) -> None:
    """Logs every pipeline execution (success AND failure) to done_so_far.md with IST timestamps."""
    block = (
        f"### [{_ist_now()}] — Pipeline Run: {run_id}\n"
        f"- Status: {status}\n"
        f"- Stage Timings (s): {json.dumps(stage_timings, indent=2, sort_keys=True)}\n"
        f"- Mesh Report: {json.dumps(mesh_report, indent=2, sort_keys=True)}\n"
        f"- Notes: {notes or 'None'}\n"
        f"---\n"
    )
    _append_markdown(DONE_SO_FAR_PATH, block)


if __name__ == "__main__":
    # Self-verify Step 0 and write its own audit entry.
    log_feature_progress(
        feature_name="Step 0 — Continuous Audit Logger & Role Tracker",
        role_owner="Role C (Docs & Audit)",
        role_importance_and_job=(
            "Secures Documentation (15 pts) & Reproducibility (15 pts): records every coded "
            "feature, role job, and pipeline run (success and failure) with IST timestamps."
        ),
        files_modified=["src/logger.py", "src/__init__.py"],
        technical_explanation=(
            "Dual appender writing structured Markdown blocks to done_so_far.md and "
            "what_we_did.md; IST computed via zoneinfo.ZoneInfo('Asia/Kolkata'), the standard-"
            "library tz database replacement for the manual UTC+5:30 timedelta."
        ),
        duration_or_latency="N/A",
    )
    print(f"Logger verified. Audit files at:\n  {DONE_SO_FAR_PATH}\n  {WHAT_WE_DID_PATH}")