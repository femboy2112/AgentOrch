"""C1 tests: Job + persistent JobQueue (pure data layer, hermetic)."""
from __future__ import annotations

import json

import pytest

from harness.job_queue import JOB_STATUSES, Job, JobQueue


@pytest.fixture()
def qpath(tmp_path):
    return tmp_path / "queue.json"


def test_job_statuses_contract():
    assert JOB_STATUSES == ("queued", "running", "done", "failed", "canceled")


def test_job_round_trip_to_from_dict():
    job = Job(id="20260101-000000-000-abcd", instruction="hi", kwargs={"mode": "direct"})
    d = job.to_dict()
    assert d["id"] == job.id and d["instruction"] == "hi" and d["kwargs"]["mode"] == "direct"
    back = Job.from_dict(d)
    assert back == job


def test_job_id_sortable_and_unique(qpath):
    q = JobQueue(qpath)
    ids = [q.submit(f"i{i}", {}, []).id for i in range(20)]
    assert len(set(ids)) == 20  # unique
    assert ids == sorted(ids)   # minted in monotonically-sortable order


def test_persistence_round_trip(qpath):
    q = JobQueue(qpath)
    a = q.submit("first", {"mode": "direct"}, ["codex"])
    b = q.submit("second", {"mode": "master"}, ["agy"])
    q.mark_running(a.id, "run-a")

    q2 = JobQueue(qpath)
    q2.load()
    ja = q2.get(a.id)
    jb = q2.get(b.id)
    assert jb is not None and jb.status == "queued" and jb.pools == ["agy"]
    # a was running -> reset to queued on reload (broker died)
    assert ja is not None and ja.status == "queued" and ja.started_at is None and ja.run_id is None


def test_atomic_write_leaves_valid_json(qpath):
    q = JobQueue(qpath)
    q.submit("x", {}, [])
    # no tmp files left behind
    leftovers = list(qpath.parent.glob(qpath.name + ".tmp*"))
    assert leftovers == []
    data = json.loads(qpath.read_text())
    assert "jobs" in data and len(data["jobs"]) == 1


def test_garbage_file_degrades_to_empty(qpath):
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text("{ this is not <<< valid json ]]")
    q = JobQueue(qpath)
    q.load()  # must not raise
    assert q.snapshot() == []


def test_missing_file_degrades_to_empty(tmp_path):
    q = JobQueue(tmp_path / "nope" / "queue.json")
    q.load()
    assert q.snapshot() == []


def test_half_written_truncated_file(qpath):
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text('{"jobs": [{"id": "20260101-000000-000-aaaa", "instr')  # truncated
    q = JobQueue(qpath)
    q.load()
    assert q.snapshot() == []


def test_running_reset_to_queued_on_reload(qpath):
    q = JobQueue(qpath)
    j = q.submit("work", {}, ["codex"])
    q.mark_running(j.id, "run-1")
    assert q.get(j.id).status == "running"

    q2 = JobQueue(qpath)
    q2.load()
    assert q2.get(j.id).status == "queued"


def test_next_runnable_fifo(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, ["codex"])
    q.submit("b", {}, ["agy"])
    # oldest queued, no active pools
    assert q.next_runnable(set()).id == a.id


def test_next_runnable_pool_disjoint_preference(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, ["codex"])      # collides with active
    b = q.submit("b", {}, ["agy"])        # disjoint
    # codex is active -> skip a, pick b
    chosen = q.next_runnable({"codex"})
    assert chosen.id == b.id
    # nothing active -> FIFO picks a
    assert q.next_runnable(set()).id == a.id


def test_next_runnable_none_when_all_collide(qpath):
    q = JobQueue(qpath)
    q.submit("a", {}, ["codex"])
    q.submit("b", {}, ["codex", "agy"])
    assert q.next_runnable({"codex"}) is None


def test_next_runnable_skips_non_queued(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, ["codex"])
    b = q.submit("b", {}, ["agy"])
    q.mark_running(a.id, "r")
    # a is running now -> next runnable is b
    assert q.next_runnable(set()).id == b.id


def test_oldest_queued_ignores_pools(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, ["codex"])
    q.submit("b", {}, ["codex"])
    assert q.oldest_queued().id == a.id


def test_mark_done_and_failed(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, [])
    b = q.submit("b", {}, [])
    q.mark_done(a.id, {"success": True, "run_id": "run-a"})
    q.mark_failed(b.id, "boom")
    assert q.get(a.id).status == "done"
    assert q.get(a.id).result["success"] is True
    assert q.get(a.id).run_id == "run-a"
    assert q.get(b.id).status == "failed"
    assert q.get(b.id).result["error"] == "boom"


def test_cancel_only_when_queued(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, [])
    assert q.cancel(a.id) is True
    assert q.get(a.id).status == "canceled"
    # cannot cancel a running job
    b = q.submit("b", {}, [])
    q.mark_running(b.id, "r")
    assert q.cancel(b.id) is False
    assert q.get(b.id).status == "running"


def test_cancel_unknown_job(qpath):
    q = JobQueue(qpath)
    assert q.cancel("does-not-exist") is False


def test_snapshot_active_first(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {}, [])
    b = q.submit("b", {}, [])
    c = q.submit("c", {}, [])
    q.mark_done(a.id, {"success": True})  # terminal
    q.mark_running(b.id, "r")             # active
    snap = q.snapshot()
    statuses = [s["status"] for s in snap]
    # active (b running, c queued) come before terminal (a done)
    assert statuses.index("done") == len(snap) - 1
    ids = [s["id"] for s in snap]
    assert set(ids) == {a.id, b.id, c.id}


def test_persisted_records_survive_full_cycle(qpath):
    q = JobQueue(qpath)
    a = q.submit("a", {"mode": "direct"}, ["codex"])
    q.mark_running(a.id, "run-a")
    q.mark_done(a.id, {"success": True, "run_id": "run-a"})
    q2 = JobQueue(qpath)
    q2.load()
    ja = q2.get(a.id)
    assert ja.status == "done" and ja.result["run_id"] == "run-a" and ja.pools == ["codex"]
