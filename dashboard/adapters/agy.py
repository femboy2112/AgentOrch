from __future__ import annotations

import re
from typing import List

_WATCHDOG_RE = re.compile(r"\[watchdog:([^\]]+)\]")


def parse_stderr_line(line: str) -> List[dict]:
    text = line.strip()
    if not text:
        return []
    out = [{"kind": "stderr", "text": text}]
    m = _WATCHDOG_RE.search(text)
    if m:
        reason = m.group(1).strip()
        out.append({"kind": "watchdog", "text": f"[watchdog:{reason}]", "data": {"reason": reason}})
    return out


def parse_stdout_text(text: str) -> List[dict]:
    body = (text or "").strip()
    if not body:
        return []
    return [{"kind": "message", "text": body}]
