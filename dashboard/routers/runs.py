from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter()


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)


def _preview_from_prompt(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "## Instruction"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1].strip()
    line = tail.splitlines()[0].strip() if tail else ""
    return line[:200]


def _started_at(run_id: str) -> Optional[float]:
    try:
        # run_ids use truncated micros (e.g. -123); pad to 6 digits for %f
        if run_id.count("-") == 2:
            prefix, micros = run_id.rsplit("-", 1)
            micros = (micros + "000000")[:6]
            run_id = f"{prefix}-{micros}"
        stamp = dt.datetime.strptime(run_id, "%Y%m%d-%H%M%S-%f")
        return stamp.timestamp()
    except Exception:
        return None


def _to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    val = value.strip().lower()
    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False
    return None


@router.get("/runs")
def list_runs(
    request: Request,
    limit: int = 50,
    before: Optional[str] = None,
    q: Optional[str] = None,
    mode: Optional[str] = None,
    success: Optional[str] = None,
    worker: Optional[str] = None,
) -> dict:
    state = request.app.state.dashboard
    rows = _run_dirs(state.runs_dir)

    if before:
        for idx, path in enumerate(rows):
            if path.name == before:
                rows = rows[idx + 1 :]
                break

    mode_filter = mode.strip().lower() if mode else None
    worker_filter = worker.strip().lower() if worker else None
    q_filter = q.strip().lower() if q else None
    success_filter = _to_bool(success)

    out: list[dict] = []
    for run_dir in rows:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_mode = str(meta.get("mode") or "")
        run_success = bool(meta.get("success"))
        generator = str(meta.get("generator") or "")
        critic = meta.get("critic")
        instruction_preview = _preview_from_prompt(run_dir / "prompt.txt")

        if mode_filter and run_mode.lower() != mode_filter:
            continue
        if success_filter is not None and run_success != success_filter:
            continue
        if worker_filter and worker_filter not in f"{generator} {critic or ''}".lower():
            continue
        if q_filter and q_filter not in (
            f"{run_dir.name} {run_mode} {generator} {critic or ''} {instruction_preview}".lower()
        ):
            continue

        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        out.append(
            {
                "run_id": run_dir.name,
                "mode": run_mode,
                "generator": generator,
                "critic": critic,
                "duration_s": float(meta.get("duration_s") or 0.0),
                "success": run_success,
                "quality": quality.get("confidence"),
                "changed_files_count": len(meta.get("changed_files") or []),
                "instruction_preview": instruction_preview,
                "started_at": _started_at(run_dir.name),
            }
        )
        if len(out) >= max(1, min(limit, 200)):
            break

    next_before = out[-1]["run_id"] if len(out) == max(1, min(limit, 200)) else None
    return {"runs": out, "next_before": next_before}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    state = request.app.state.dashboard
    run_dir = state.runs_dir / run_id
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="run not found")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to parse meta.json: {exc}") from exc

    artifacts = {
        "prompt": str(run_dir / "prompt.txt"),
        "stdout": str(run_dir / "stdout.log"),
        "stderr": str(run_dir / "stderr.log"),
        "diff": str(run_dir / "changed-files.diff"),
        "events": str(run_dir / "events.jsonl"),
    }
    return {"meta": meta, "artifacts": artifacts}


_ARTIFACTS = {
    "prompt": "prompt.txt",
    "stdout": "stdout.log",
    "stderr": "stderr.log",
    "diff": "changed-files.diff",
    "events": "events.jsonl",
}


def _read_artifact(path: Path, range_header: Optional[str]) -> tuple[int, bytes, dict]:
    payload = path.read_bytes()
    headers = {"Accept-Ranges": "bytes"}
    if not range_header or not range_header.startswith("bytes="):
        return 200, payload, headers

    try:
        start_end = range_header.split("=", 1)[1]
        start_str, end_str = start_end.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else len(payload) - 1
    except Exception:
        return 200, payload, headers

    if start < 0:
        start = 0
    if end >= len(payload):
        end = len(payload) - 1
    if start > end or start >= len(payload):
        return 416, b"", headers

    chunk = payload[start : end + 1]
    headers["Content-Range"] = f"bytes {start}-{end}/{len(payload)}"
    return 206, chunk, headers


@router.get("/runs/{run_id}/artifact/{name}")
def get_artifact(run_id: str, name: str, request: Request) -> Response:
    state = request.app.state.dashboard
    file_name = _ARTIFACTS.get(name)
    if file_name is None:
        raise HTTPException(status_code=404, detail="unknown artifact")

    path = state.runs_dir / run_id / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    status, payload, headers = _read_artifact(path, request.headers.get("range"))
    media_type = "application/x-ndjson" if name == "events" else "text/plain; charset=utf-8"
    return Response(content=payload, status_code=status, media_type=media_type, headers=headers)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, request: Request) -> dict:
    state = request.app.state.dashboard
    run_dir = state.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")

    for path in sorted(run_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            path.rmdir()
    run_dir.rmdir()
    return {"deleted": True}
