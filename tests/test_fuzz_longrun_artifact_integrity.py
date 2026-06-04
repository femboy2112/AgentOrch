"""Fuzz / long-run resilience probes — DIMENSION: run-artifact / state write integrity.

These probes simulate the conditions that only bite on long / mission-critical runs
whose ONLY durable record is the per-run artifact set (meta.json, changed-files.diff,
events.jsonl) and the before/after snapshot diff. Each probe asserts the DESIRED
RESILIENT behavior; probes that fail today (real defects) are decorated strict-xfail
so the committed suite stays green and flips to a failure (XPASS) once fixed.

Hermetic + fast: no live worker, no real build, tiny tmp_path fixtures, milliseconds.

The artifact-write tail (stdout.log / changed-files.diff / meta.json) and the
events.jsonl `_sink` closure live INLINE inside harness.dispatch.dispatch() — they
are not separately importable, so each probe reproduces the EXACT current code line
(quoted from harness/dispatch.py) against tmp_path and asserts on the result. The
quoted line is the contract under test; if the source line changes, update the quote.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.snapshot import take_snapshot, diff_snapshots, MAX_DIFF_BYTES
from harness.dispatch import _atomic_write_text


# A string carrying a lone UTF-16 surrogate — non-encodable as strict utf-8.
# This is what a worker reply / log line assembled from surrogateescape bytes,
# or a model emitting a stray codepoint, lands in `output` as.
SURROGATE = "build ok \udce9 done"


# ---------------------------------------------------------------------------
# Probe 1 — surrogate in worker output crashes the final artifact-write tail.
#   harness/dispatch.py:2107  (run_dir/'stdout.log').write_text(output, encoding='utf-8')
#   harness/dispatch.py:2262  (run_dir/'meta.json').write_text(json.dumps(meta_dict, indent=2), 'utf-8')
# These writes are OUTSIDE the try/except, so the exception propagates out of
# dispatch(): a SUCCESSFUL build produces NO meta.json and looks failed.
# ---------------------------------------------------------------------------
@pytest.mark.not_slow
def test_surrogate_output_does_not_crash_artifact_tail(tmp_path):
    # The fix routes every artifact write through harness.dispatch._atomic_write_text
    # (errors="backslashreplace"), so a surrogate/non-encodable char in worker output
    # degrades gracefully instead of raising UnicodeEncodeError out of the unguarded
    # final tail and leaving a successful run with NO meta.json.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = SURROGATE  # the worker's assembled stdout

    # Must not raise (was UnicodeEncodeError on a plain write_text(encoding='utf-8')).
    _atomic_write_text(run_dir / "stdout.log", output)

    meta_dict = {"success": True, "tail_note": output}
    _atomic_write_text(run_dir / "meta.json", json.dumps(meta_dict, indent=2))

    assert (run_dir / "meta.json").exists()
    reloaded = json.loads((run_dir / "meta.json").read_text(encoding="utf-8", errors="replace"))
    assert reloaded["success"] is True
    # No stray temp left behind once the write completes.
    assert not list(run_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Probe 2 — non-atomic artifact writes leave a truncated meta.json on a mid-write
# kill (SIGKILL/OOM/scope exit-144/ENOSPC). master.py:637-640 uses tmp+os.replace
# ("atomic: crash mid-write can't corrupt"); the dispatch artifacts at 2262 do not.
# DESIRED: a reader of meta.json sees it valid-or-absent, never half-written.
# ---------------------------------------------------------------------------
@pytest.mark.not_slow
def test_meta_json_write_is_atomic_under_midwrite_kill(tmp_path, monkeypatch):
    # _atomic_write_text writes a sibling .tmp then publishes with os.replace, so an
    # interruption at the publish boundary leaves the TARGET untouched (complete-old
    # or absent), never a half-written JSON. Prove the guarantee: an existing valid
    # meta.json must survive an overwrite whose os.replace is interrupted (kill/ENOSPC).
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta_path = run_dir / "meta.json"

    v1 = json.dumps({"success": True, "rev": 1}, indent=2)
    _atomic_write_text(meta_path, v1)
    assert json.loads(meta_path.read_text(encoding="utf-8")) == {"success": True, "rev": 1}

    # Simulate the process dying exactly at the publish step of a second write.
    def _boom(*a, **k):
        raise OSError("killed mid-publish (ENOSPC/SIGKILL)")
    monkeypatch.setattr(os, "replace", _boom)

    v2 = json.dumps({"success": True, "rev": 2, "tokens": {}}, indent=2)
    with pytest.raises(OSError):
        _atomic_write_text(meta_path, v2)

    # The target is NEVER a partial JSON: it still holds the complete v1, fully
    # parseable. (A plain write_text would have truncated it to half of v2.)
    on_disk = meta_path.read_text(encoding="utf-8")
    assert json.loads(on_disk) == {"success": True, "rev": 1}


# ---------------------------------------------------------------------------
# Probe 3 — take_snapshot reads every file FULLY into memory to hash it
# (snapshot.py:64  blob = fpath.read_bytes()), even huge binaries whose content is
# never retained (MAX_DIFF_BYTES at line 70 only gates retention, not the read).
# Snapshotting the out-dir before+after every dispatch => a multi-GB generated
# artifact is materialized in RAM twice per run -> OOM during finalization.
# DESIRED: the per-file read is bounded/chunked, not a single read_bytes of the
# whole file.
# ---------------------------------------------------------------------------
@pytest.mark.not_slow
def test_take_snapshot_does_not_read_whole_file_into_memory(tmp_path, monkeypatch):
    # A file modestly larger than MAX_DIFF_BYTES: its content is NEVER retained, so
    # there is no functional reason to read it whole. Keep it small for speed; the
    # defect is structural (full read regardless of size).
    big = tmp_path / "artifact.bin"
    big.write_bytes(b"\x00\x01" * (MAX_DIFF_BYTES))  # ~2 MB, binary, > MAX_DIFF_BYTES

    biggest_single_read = {"n": 0}
    real_read_bytes = Path.read_bytes

    def spy_read_bytes(self):
        data = real_read_bytes(self)
        if self == big or self.name == "artifact.bin":
            biggest_single_read["n"] = max(biggest_single_read["n"], len(data))
        return data

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)

    snap = take_snapshot(tmp_path)
    assert "artifact.bin" in snap.files  # still hashed/recorded

    # DESIRED: the hash is computed by streaming bounded chunks, so no single read
    # ever materializes the whole multi-MB file. A bounded reader would cap each
    # allocation at a small block (e.g. <= 1 MiB). RED today: read_bytes pulls the
    # entire ~2 MB file in one allocation.
    assert biggest_single_read["n"] <= 1024 * 1024, (
        f"snapshot read {biggest_single_read['n']} bytes in a single allocation; "
        "expected bounded/chunked hashing"
    )


# ---------------------------------------------------------------------------
# Probe 4 — events.jsonl silently loses an event line when an event carries a
# non-encodable char. dispatch.py:1609-1610 _sink: json.dumps(ensure_ascii=False)
# + fh.write to a utf-8 file. A surrogate in the payload raises UnicodeEncodeError;
# EventBus.publish swallows sink exceptions (event_bus.py:137-141), so the run
# survives but that event LINE is dropped — under-counting tokens / gapping replay.
# DESIRED: every published event lands in events.jsonl.
# ---------------------------------------------------------------------------
@pytest.mark.not_slow
@pytest.mark.xfail(
    reason="fuzz-longrun artifact_integrity: dispatch.py:1609 _sink does json.dumps+write to a "
    "utf-8 file; a surrogate-bearing event raises UnicodeEncodeError (swallowed upstream) so the "
    "event line is silently dropped from events.jsonl",
    strict=True,
)
def test_events_jsonl_sink_does_not_drop_surrogate_event(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.touch()

    # Reproduce the EXACT inline _sink write (harness/dispatch.py:1609-1610).
    def _sink(event: dict) -> None:
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    fed = [
        {"kind": "usage", "data": {"usage_kind": "call"}, "msg": "clean"},
        {"kind": "log", "msg": SURROGATE},  # surrogate in a log line echoing worker output
        {"kind": "usage", "data": {"usage_kind": "call"}, "msg": "after"},
    ]
    for ev in fed:
        # The real call path (EventBus.publish) swallows the sink exception; model
        # that so the "run survives" but the line is lost.
        try:
            _sink(ev)
        except Exception:
            pass

    lines = [ln for ln in events_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    # DESIRED: every fed event is durably recorded (sanitize / surrogatepass before write).
    assert len(lines) == len(fed), (
        f"events.jsonl has {len(lines)} lines, fed {len(fed)} — a surrogate event was silently dropped"
    )


# ---------------------------------------------------------------------------
# Probe 5 — an unreadable file (chmod 000 after the worker ran) that existed in the
# before-snapshot is reported as DELETED by diff_snapshots (snapshot.py:65-66 skips
# it on OSError; 107-109 before-minus-after => false delete). The --protect-paths
# gate trusts the diff path list, so an unreadable protected file's status is
# misrepresented. DESIRED: the diff distinguishes "unreadable" from "deleted".
# ---------------------------------------------------------------------------
@pytest.mark.not_slow
@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 000 is bypassed for root; can't make file unreadable")
def test_snapshot_distinguishes_unreadable_from_deleted(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("important content the worker should not be able to delete\n")

    before = take_snapshot(tmp_path)
    assert "secret.txt" in before.files

    # Worker makes it unreadable (still on disk, NOT deleted).
    os.chmod(f, 0o000)
    try:
        after = take_snapshot(tmp_path)
        diff = diff_snapshots(before, after)

        # The file still EXISTS — reporting it as a deletion is wrong and would let a
        # --protect-paths reviewer believe a protected file was removed (or miss that
        # it became unreadable). DESIRED: it is NOT in `deleted`; instead the snapshot
        # carries an explicit unreadable marker (e.g. still present in after.files, or
        # surfaced via a dedicated unreadable/error class).
        assert "secret.txt" not in diff.deleted, (
            "an unreadable-but-present file is reported as DELETED (false delete)"
        )
        assert "secret.txt" in after.files, (
            "an unreadable file is silently omitted from the snapshot with no marker"
        )
    finally:
        os.chmod(f, 0o644)  # restore so tmp_path cleanup works
