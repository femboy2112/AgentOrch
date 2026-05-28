from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AdapterCtx:
    state: Dict[str, Any] = field(default_factory=dict)


def load_json_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None
