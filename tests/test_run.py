import json
import sqlite3

import pytest

from aqg.contract import load_template
from aqg.endpoints.client import Completion, EndpointError
from aqg.endpoints.registry import load_registry
from aqg.run import RunLockedError, Runner, StudyConfig, call_seed, plan_cells, prepare, run_lock, status_rows
from aqg.store import Store, StudyMismatchError
from aqg.topics import Topic

REGISTRY = load_registry()
TEMPLATE = load_template()
TEXT = "### Closed loop\nControl action depends on the system output (feedback). It corrects disturbances."
TOPICS = [Topic("SYSTEM-2022-1", "SYSTEM", "Basics", TEXT), Topic("SYSTEM-2022-2", "SYSTEM", "Models", TEXT)]
GOOD = json.dumps({"questions": [{
    "question": "Which property of a closed-loop control system allows it to correct the effect of disturbances "
                "on the output?",
    "options": {"A": "Feedback of the measured output", "B": "A larger actuator gain",
                "C": "Feedforward of the setpoint value", "D": "An open-loop reference model"},
    "answer": "A", "evidence": "Control action depends on the system output (feedback)."}] * 5})


def study(**changes) -> StudyConfig:
    values = {"study_id": "test", "endpoints": ["gpt-oss", "cohere"], "runs": 2, "seed": 7,
              "topics_dir": "unused", "database": "unused", "budget_usd": 1.0, "max_attempts": 3}
    return StudyConfig(**{**values, **changes})


class FakeClient:
    """Answers each call with the next scripted reply for its endpoint: a text, or an exception to raise."""

    def __init__(self, spec, script, sent):
        self.spec, self.script, self.sent = spec, script, sent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def complete(self, messages, overrides=None):
        self.sent.append((self.spec.alias, overrides))
        reply = self.script[self.spec.alias].pop(0) if self.script[self.spec.alias] else GOOD
        if isinstance(reply, Exception):
            raise reply
        return Completion(alias=self.spec.alias, provider=self.spec.provider, host=self.spec.host,
                          requested_model_id=self.spec.model_id, returned_model_id=self.spec.model_id,
                          served_by=self.spec.host, text=reply, finish_reason="stop", prompt_tokens=1000,
                          completion_tokens=800, reasoning_tokens=None, billed_prompt_tokens=None,
                          billed_completion_tokens=None, billed_cost_usd=None, estimated_cost_usd=0.001,
                          latency_s=1.5, elapsed_s=1.6, attempts=1, started_at="2026-10-28T00:00:00+00:00",
                          settings={**self.spec.settings, **(overrides or {})},
                          request_body={"messages": messages}, response_body={"ok": True})


def run(config, store, script=None, sent=None, topics=TOPICS):
    script = {alias: list(replies) for alias, replies in (script or {}).items()}
    for alias in config.endpoints:
        script.setdefault(alias, [])
    sent = [] if sent is None else sent
    cells = prepare(config, store, topics, REGISTRY, TEMPLATE, "abc123")
    runner = Runner(config, store, topics, cells, REGISTRY, TEMPLATE,
                    client_factory=lambda spec: FakeClient(spec, script, sent), log=lambda *_: None, error_pause_s=0)
    runner.run()
    return runner, cells


def test_plan_is_complete_seeded_and_interleaved():
    cells = plan_cells(study(), ["SYSTEM-2022-1", "SYSTEM-2022-2"])
    assert len(cells) == 8 and len({c.cell_id for c in cells}) == 8
    assert [c.cell_id for c in cells] == [c.cell_id for c in plan_cells(study(), ["SYSTEM-2022-1", "SYSTEM-2022-2"])]
    assert [c.cell_id for c in cells] != [c.cell_id for c in plan_cells(study(seed=8), ["SYSTEM-2022-1", "SYSTEM-2022-2"])]


def test_every_cell_is_done_and_items_are_balanced_and_scored():
    store = Store(":memory:")
    run(study(), store)
    assert store.query("SELECT COUNT(*) FROM call WHERE status = 'accepted'")[0][0] == 8
    items = store.query("SELECT * FROM item")
    assert len(items) == 40
    assert {r["answer"] for r in items} == {"A", "B", "C", "D"} and {r["raw_answer"] for r in items} == {"A"}
    assert all(r["evidence_score"] == 100 for r in items)
    assert all(json.loads(r["flags_json"]) == [] for r in items)
    row = store.query("SELECT * FROM call LIMIT 1")[0]
    assert row["json_strict"] == 1 and row["reply_strict"] == 1 and row["parse_mode"] == "strict"


def test_a_rejected_reply_is_recorded_and_retried_then_resume_calls_nothing():
    store, sent = Store(":memory:"), []
    run(study(endpoints=["gpt-oss"], runs=1, topics=None), store, {"gpt-oss": ["not json at all"]}, sent)
    calls = store.query("SELECT cell_id, attempt, status, errors_json FROM call ORDER BY call_id")
    assert [r["status"] for r in calls] == ["rejected", "accepted", "accepted"]
    assert calls[0]["cell_id"] == calls[1]["cell_id"] and calls[1]["attempt"] == 2
    assert "parse" in json.loads(calls[0]["errors_json"])[0]
    before = len(sent)
    run(study(endpoints=["gpt-oss"], runs=1, topics=None), store, sent=sent)
    assert len(sent) == before                                       # resumed: every cell was done


def test_seeds_differ_by_run_and_attempt_and_are_sent():
    sent = []
    run(study(endpoints=["gpt-oss"], runs=2), Store(":memory:"), {"gpt-oss": ["bad"]}, sent)
    seeds = [overrides["seed"] for _, overrides in sent]
    assert len(set(seeds)) == len(seeds) == 5
    assert call_seed(7, "SYSTEM-2022-1|gpt-oss|r1", 1) == call_seed(7, "SYSTEM-2022-1|gpt-oss|r1", 1)


def test_attempts_are_capped_and_the_cell_counts_as_failed():
    store = Store(":memory:")
    run(study(endpoints=["gpt-oss"], runs=1), store, {"gpt-oss": ["bad"] * 3})
    first = store.query("SELECT cell_id, COUNT(*) AS n FROM call GROUP BY cell_id ORDER BY MIN(call_id)")[0]
    assert first["n"] == 3
    assert status_rows(store, 3)[0]["failed"] == 1 and status_rows(store, 3)[0]["done"] == 1


def test_an_account_error_stops_only_that_endpoint():
    store = Store(":memory:")
    error = EndpointError("cohere: HTTP 401: invalid api token", "cohere", 401, 1, [{"attempt": 1, "status": 401}])
    runner, _ = run(study(), store, {"cohere": [error]})
    assert "cohere" in runner.stopped_endpoints and runner.stop_reason is None
    rows = {r["endpoint"]: r for r in status_rows(store, 3)}
    assert rows["gpt-oss"]["done"] == 4 and rows["cohere"]["done"] == 0 and rows["cohere"]["errors"] == 1


def test_a_temporary_error_is_recorded_and_the_cell_tried_again():
    store = Store(":memory:")
    error = EndpointError("gpt-oss: failed after 4 attempts", "gpt-oss", 503, 4, [{"attempt": 4, "status": 503}])
    run(study(endpoints=["gpt-oss"]), store, {"gpt-oss": [error]})
    assert [r["status"] for r in store.query("SELECT status FROM call ORDER BY call_id")][:2] == ["error", "accepted"]


def test_the_budget_stops_the_run():
    store = Store(":memory:")
    runner, _ = run(study(endpoints=["gpt-oss"], budget_usd=0.0025), store)
    assert store.query("SELECT COUNT(*) FROM call")[0][0] == 3 and "budget" in runner.stop_reason


def test_a_resumed_study_must_be_the_same():
    store = Store(":memory:")
    run(study(), store)
    with pytest.raises(StudyMismatchError, match="settings changed"):
        prepare(study(runs=3), store, TOPICS, REGISTRY, TEMPLATE, "abc123")
    changed = [Topic("SYSTEM-2022-1", "SYSTEM", "Basics", TEXT + " Edited."), TOPICS[1]]
    with pytest.raises(StudyMismatchError, match="SYSTEM-2022-1"):
        prepare(study(), store, changed, REGISTRY, TEMPLATE, "abc123")
    with pytest.raises(StudyMismatchError, match="cells differ"):
        prepare(study(), store, TOPICS[:1], REGISTRY, TEMPLATE, "abc123")
    # operational settings may change between sessions
    prepare(study(budget_usd=5.0, workers_per_endpoint=2), store, TOPICS, REGISTRY, TEMPLATE, "abc123")


def test_a_used_up_monthly_quota_stops_that_endpoint():
    store = Store(":memory:")
    error = EndpointError("cohere: failed after 4 attempts; last: HTTP 429: You are using a Trial key, which is "
                          "limited to 1000 API calls / month.", "cohere", 429, 4, [{"attempt": 4, "status": 429}])
    runner, _ = run(study(), store, {"cohere": [error]})
    assert "cohere" in runner.stopped_endpoints
    assert {r["endpoint"]: r for r in status_rows(store, 3)}["cohere"]["calls"] == 1


def test_an_unexpected_error_stops_the_endpoint_instead_of_killing_the_worker_silently():
    store = Store(":memory:")
    runner, _ = run(study(), store, {"cohere": [RuntimeError("socket exploded")]})
    assert "socket exploded" in runner.stopped_endpoints["cohere"]
    assert {r["endpoint"]: r for r in status_rows(store, 3)}["gpt-oss"]["done"] == 4


def test_only_one_run_at_a_time_per_database(tmp_path):
    database = tmp_path / "study.sqlite"
    with run_lock(database):
        with pytest.raises(RunLockedError, match="another run"):
            with run_lock(database):
                pass
    with run_lock(database):          # released afterwards, also after an error
        pass
    assert not (tmp_path / "study.sqlite.lock").exists()


def test_calls_and_items_cannot_be_changed_or_deleted():
    store = Store(":memory:")
    run(study(endpoints=["gpt-oss"], runs=1), store)
    for sql in ("UPDATE call SET status = 'rejected'", "DELETE FROM call", "UPDATE item SET answer = 'A'",
                "DELETE FROM item"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store.query(sql)
