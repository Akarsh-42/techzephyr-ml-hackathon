"""Continuous progress tracker — appends timestamped entries to done_so_far.md.
Every dev/agent MUST call this after writing/updating any module or milestone.
This file later becomes source material for README.md (15 rubric points)."""
from datetime import datetime
from config import LOG_FILE

_HEADER = "# done_so_far — SKETCH2SHAPE AI engineering changelog\n\n"

_TEMPLATE = """### [{ts}] — Feature: {name}
- **Author / Agent:** {author}
- **Files Created/Updated:** `{files}`
- **Time Taken / Execution Time:** {dur}
- **Technical Explanation:** {desc}
---

"""


def log_feature_progress(feature_name: str, files_modified: list, description: str,
                          duration_or_latency: str = "N/A",
                          author_or_agent: str = "Dev/Agent") -> None:
    if not LOG_FILE.exists():
        LOG_FILE.write_text(_HEADER, encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = _TEMPLATE.format(ts=ts, name=feature_name, author=author_or_agent,
                             files=", ".join(files_modified), dur=duration_or_latency,
                             desc=description)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(block)