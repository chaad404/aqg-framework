"""Runs a study: every topic x endpoint x run cell, resumable, with every call attempt recorded.

    python -m aqg.run configs/pilot.yaml --plan      # the cells and a rough cost estimate; calls nothing
    python -m aqg.run configs/pilot.yaml             # run the study, or resume it
    python -m aqg.run configs/pilot.yaml --status    # progress, read from the database

Each endpoint has its own worker and works through its cells in a seeded random order, so every endpoint's
calls are spread over the whole run and topics and runs are mixed in time. A cell is done when one reply is
accepted. A rejected reply or a failed request is recorded and the cell is tried again, up to `max_attempts`
times. Request or account problems (HTTP 400-404) stop that endpoint; the budget stops the whole run.
"""
import argparse
import hashlib
import os
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from aqg import config
from aqg.contract import DEFAULT_TEMPLATE, ITEMS_PER_CALL, build_messages, load_template, prompt_sha256
from aqg.endpoints.client import EndpointClient, EndpointError
from aqg.endpoints.registry import get_registry
from aqg.store import Store, StudyMismatchError, dumps, now
from aqg.topics import TopicFileError, load_topics
from aqg.validate import balance_keys, check_reply, evidence_score, item_flags

STOP_STATUSES = {400, 401, 402, 403, 404}
MONTHLY_QUOTA = re.compile(r"/\s*month|per month|monthly", re.I)   # e.g. Cohere's trial key: waiting cannot help
# Settings that only steer how a run is carried out; changing them does not change the study
OPERATIONAL = {"topics_dir", "database", "workers_per_endpoint", "budget_usd", "require_clean_code"}
# Rough output per call for the --plan estimate: 5 items with evidence, plus gpt-oss reasoning
EST_OUTPUT_TOKENS, EST_REASONING_TOKENS, CHARS_PER_TOKEN = 2000, 1500, 4


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    endpoints: list[str] = Field(min_length=1)
    runs: int = Field(ge=1)
    topics: list[str] | None = None        # None: every topic file in topics_dir
    seed: int
    template: str = DEFAULT_TEMPLATE
    items_per_call: int = Field(default=ITEMS_PER_CALL, ge=1)
    max_attempts: int = Field(default=4, ge=1)        # the first call and up to 3 retries
    send_seed: bool = True
    topics_dir: str                          # relative to the repository root
    database: str                            # relative to the repository root
    workers_per_endpoint: int = Field(default=1, ge=1)
    budget_usd: float = Field(gt=0)
    require_clean_code: bool = False         # the main run must come from a committed, tagged state

    @property
    def identity(self) -> dict:
        return self.model_dump(exclude=OPERATIONAL)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(dumps(self.identity).encode("utf-8")).hexdigest()

    def path(self, value: str) -> Path:
        return (config.REPO_ROOT / value).resolve()


def load_study(path: Path | str) -> StudyConfig:
    return StudyConfig(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))


@dataclass(frozen=True)
class Cell:
    cell_id: str
    topic_id: str
    endpoint: str
    run_no: int
    order_no: int


def plan_cells(study: StudyConfig, topic_ids: list[str]) -> list[Cell]:
    keys = [(t, e, r) for t in topic_ids for e in study.endpoints for r in range(1, study.runs + 1)]
    random.Random(study.seed).shuffle(keys)
    return [Cell(f"{t}|{e}|r{r}", t, e, r, n) for n, (t, e, r) in enumerate(keys)]


def call_seed(study_seed: int, cell_id: str, attempt: int) -> int:
    """A different seed for every run and attempt, so runs are independent samples but can be repeated."""
    return int(hashlib.sha256(f"{study_seed}|{cell_id}|{attempt}".encode()).hexdigest()[:8], 16)


def code_version() -> str | None:
    def git(*args) -> str:
        return subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}+dirty" if commit and dirty else commit or None


class RunLockedError(RuntimeError):
    pass


@contextmanager
def run_lock(database: Path):
    """Only one run per database at a time: two would repeat each other's cells."""
    lock = database.with_name(database.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunLockedError(f"another run is using {database.name}. If none is running (for example after a "
                             f"crash), delete {lock} and start again.") from None
    try:
        os.write(fd, f"pid {os.getpid()}, started {now()}\n".encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def prepare(study: StudyConfig, store: Store, topics: list, registry, template, version: str | None) -> list[Cell]:
    """Records a new study or checks a resumed one; returns its cells."""
    cells = plan_cells(study, [t.topic_id for t in topics])
    endpoints = {alias: registry.get(alias).model_dump() for alias in study.endpoints}
    new = store.open_study(study.study_id, study.identity, study.sha256, template, endpoints,
                           registry.pricing_date, version)
    stored = {r["cell_id"] for r in store.query("SELECT cell_id FROM cell")}
    if not new and stored != {c.cell_id for c in cells}:
        raise StudyMismatchError("The topics or cells differ from the ones this study started with.")
    store.add_topics(topics)
    store.add_cells(cells)
    return cells


class Runner:
    def __init__(self, study: StudyConfig, store: Store, topics: list, cells: list[Cell], registry, template,
                 client_factory=EndpointClient, log=print, error_pause_s: float = 30.0):
        self.study, self.store, self.registry, self.template = study, store, registry, template
        self.topics = {t.topic_id: t for t in topics}
        self.cells = cells
        self.client_factory, self.log = client_factory, log
        self.error_pause_s = error_pause_s     # after a request failed all its retries, before the next attempt
        self.stop = threading.Event()
        self.stop_reason: str | None = None
        self.stopped_endpoints: dict[str, str] = {}
        self._lock = threading.Lock()

    def run(self) -> None:
        threads = []
        for alias in self.study.endpoints:
            pending = queue.Queue()
            for cell in sorted((c for c in self.cells if c.endpoint == alias), key=lambda c: c.order_no):
                if not self.store.is_done(cell.cell_id):
                    pending.put(cell)
            spec = self.registry.get(alias)
            for _ in range(self.study.workers_per_endpoint):
                thread = threading.Thread(target=self._worker, args=(spec, pending), name=alias, daemon=True)
                thread.start()
                threads.append(thread)
        try:
            while any(t.is_alive() for t in threads):
                for thread in threads:
                    thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self._halt("stopped with Ctrl+C; the calls in progress are finished and recorded first")
            for thread in threads:
                thread.join()

    def _halt(self, reason: str) -> None:
        with self._lock:
            if not self.stop.is_set():
                self.stop_reason = reason
                self.stop.set()
                self.log(f"STOP: {reason}")

    def _stop_endpoint(self, alias: str, reason: str) -> None:
        with self._lock:
            self.stopped_endpoints[alias] = reason
        self.log(f"STOP {alias}: {reason}")

    def _worker(self, spec, pending: queue.Queue) -> None:
        try:
            client = self.client_factory(spec)
        except Exception as exc:            # for example a missing API key
            self._stop_endpoint(spec.alias, f"{type(exc).__name__}: {exc}")
            return
        with client:
            while not self.stop.is_set() and spec.alias not in self.stopped_endpoints:
                try:
                    cell = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._run_cell(client, spec, cell)
                except Exception as exc:    # never let a worker die silently: stop the endpoint and say why
                    self._stop_endpoint(spec.alias, f"unexpected error in {cell.cell_id}: {exc!r}")
                    return

    def _run_cell(self, client, spec, cell: Cell) -> None:
        topic = self.topics[cell.topic_id]
        messages = build_messages(self.template, topic.text, self.study.items_per_call)
        digest = prompt_sha256(messages)
        while not self.stop.is_set():
            attempt = self.store.attempts(cell.cell_id) + 1
            if attempt > self.study.max_attempts:
                self.log(f"{cell.cell_id}: no accepted reply in {self.study.max_attempts} attempts; counted as failed")
                return
            if self.store.spent_usd() >= self.study.budget_usd:
                self._halt(f"the budget of ${self.study.budget_usd:.2f} is used up")
                return
            seed = call_seed(self.study.seed, cell.cell_id, attempt) if self.study.send_seed else None
            outcome = self._attempt(client, spec, cell, topic, messages, digest, attempt, seed)
            if outcome in ("accepted", "stop"):
                return
            if outcome == "error":
                self.stop.wait(self.error_pause_s)     # the host is struggling: give it a moment

    def _attempt(self, client, spec, cell, topic, messages, digest, attempt, seed) -> str:
        base = {"cell_id": cell.cell_id, "attempt": attempt, "started_at": now(), "endpoint": spec.alias,
                "provider": spec.provider, "host": spec.host, "requested_model_id": spec.model_id,
                "seed": seed, "prompt_sha256": digest}
        try:
            c = client.complete(messages, overrides={"seed": seed} if seed is not None else None)
        except EndpointError as exc:
            self.store.record_call({**base, "status": "error", "finished_at": now(),
                                    "settings_json": dumps({**spec.settings, **({"seed": seed} if seed is not None else {})}),
                                    "http_attempts": exc.attempts, "retry_log_json": dumps(exc.retry_log),
                                    "errors_json": dumps([str(exc)])})
            self.log(f"{_stamp()} {spec.alias:<13} {cell.topic_id:<16} r{cell.run_no} try {attempt}: error: {exc}")
            if (exc.status in STOP_STATUSES or "activate paid use" in str(exc)
                    or (exc.status == 429 and MONTHLY_QUOTA.search(str(exc)))):
                self._stop_endpoint(spec.alias, str(exc))
                return "stop"
            return "error"

        call = {**base, "finished_at": now(), "returned_model_id": c.returned_model_id, "served_by": c.served_by,
                "settings_json": dumps(c.settings), "http_attempts": c.attempts, "retry_log_json": dumps(c.retry_log),
                "latency_s": c.latency_s, "elapsed_s": c.elapsed_s, "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens, "reasoning_tokens": c.reasoning_tokens,
                "billed_prompt_tokens": c.billed_prompt_tokens, "billed_completion_tokens": c.billed_completion_tokens,
                "billed_cost_usd": c.billed_cost_usd, "estimated_cost_usd": c.estimated_cost_usd,
                "finish_reason": c.finish_reason, "reply_text": c.text, "request_json": dumps(c.request_body),
                "response_json": dumps(c.response_body)}
        try:
            check = check_reply(c.text, self.study.items_per_call)
            call.update({"json_strict": int(check.parse.strict_ok), "parse_mode": check.parse.mode,
                         "reply_strict": int(check.strict_ok), "errors_json": dumps(check.errors),
                         "repairs_json": dumps({"parse": check.parse.repairs, "batch": check.batch_repairs})})
            rows = self._item_rows(check, cell, topic) if check.accepted else []
        except Exception as exc:            # a bug in the checks: keep the reply, stop, fix, resume
            self.store.record_call({**call, "status": "rejected", "errors_json": dumps([f"internal error: {exc!r}"])})
            self._halt(f"internal error while checking {cell.cell_id}: {exc!r}")
            return "stop"
        status = "accepted" if check.accepted else "rejected"
        self.store.record_call({**call, "status": status}, rows)
        cost = c.billed_cost_usd if c.billed_cost_usd is not None else c.estimated_cost_usd
        detail = f"{len(rows)} items" if rows else "; ".join(check.errors)[:120]
        self.log(f"{_stamp()} {spec.alias:<13} {cell.topic_id:<16} r{cell.run_no} try {attempt}: {status}, "
                 f"{c.latency_s:.1f} s, ${cost or 0:.4f}, {detail}")
        return status

    def _item_rows(self, check, cell: Cell, topic) -> list[dict]:
        moved = balance_keys([i.item for i in check.items], cell.topic_id, cell.endpoint, cell.run_no)
        rows = []
        for position, (result, item) in enumerate(zip(check.items, moved)):
            rows.append({
                "position": position, "question": item["question"],
                **{f"option_{label.lower()}": item["options"][label] for label in "ABCD"},
                "answer": item["answer"], "raw_answer": item["raw_answer"], "permutation": item["permutation"],
                "evidence": item["evidence"], "strict_ok": int(result.strict_ok),
                "repairs_json": dumps(result.repairs), "flags_json": dumps(item_flags(item)),
                "evidence_score": evidence_score(item["evidence"], topic.text)})
        return rows


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


# --- reporting -----------------------------------------------------------------------------------------

def status_rows(store: Store, max_attempts: int) -> list[dict]:
    cells = store.query("""
        SELECT c.endpoint, c.cell_id, COUNT(k.call_id) AS calls, SUM(k.status = 'accepted') AS accepted,
               SUM(k.status = 'rejected') AS rejected, SUM(k.status = 'error') AS errors,
               SUM(COALESCE(k.billed_cost_usd, k.estimated_cost_usd)) AS cost
        FROM cell c LEFT JOIN call k ON k.cell_id = c.cell_id GROUP BY c.cell_id""")
    latencies = store.query("SELECT endpoint, latency_s FROM call WHERE status = 'accepted'")
    out = {}
    for r in cells:
        row = out.setdefault(r["endpoint"], {"endpoint": r["endpoint"], "cells": 0, "done": 0, "failed": 0,
                                             "calls": 0, "rejected": 0, "errors": 0, "cost_usd": 0.0})
        row["cells"] += 1
        row["done"] += bool(r["accepted"])
        row["failed"] += not r["accepted"] and r["calls"] >= max_attempts
        row["calls"] += r["calls"]
        row["rejected"] += r["rejected"] or 0
        row["errors"] += r["errors"] or 0
        row["cost_usd"] += r["cost"] or 0.0
    for row in out.values():
        values = [r["latency_s"] for r in latencies if r["endpoint"] == row["endpoint"]]
        row["median_latency_s"] = round(statistics.median(values), 1) if values else None
        row["cost_usd"] = round(row["cost_usd"], 4)
    return list(out.values())


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(nothing yet)")
        return
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))


def estimate(study: StudyConfig, topics: list, registry, template) -> list[dict]:
    rows = []
    for alias in study.endpoints:
        spec = registry.get(alias)
        prices = spec.price_per_m
        calls, cost = 0, 0.0
        for topic in topics:
            chars = sum(len(m["content"]) for m in build_messages(template, topic.text, study.items_per_call))
            output = EST_OUTPUT_TOKENS + (EST_REASONING_TOKENS if spec.extra.get("reasoning") else 0)
            cost += study.runs * (chars / CHARS_PER_TOKEN * prices.input + output * prices.output) / 1e6
            calls += study.runs
        rows.append({"endpoint": alias, "host": spec.host, "calls": calls, "est_cost_usd": round(cost, 4)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aqg.run", description="Run, resume or inspect a study.")
    parser.add_argument("study", help="the study file, for example configs/pilot.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="show the cells and a rough cost estimate; call nothing")
    mode.add_argument("--status", action="store_true", help="show the progress stored in the database")
    parser.add_argument("--yes", action="store_true", help="start without asking for confirmation")
    args = parser.parse_args(argv)

    study = load_study(args.study)
    registry = get_registry()
    template = load_template(study.template)
    database = study.path(study.database)

    if args.status:
        if not database.exists():
            print(f"No database yet: {database}")
            return 1
        store = Store(database)
        print_table(status_rows(store, study.max_attempts))
        print(f"spent so far: ${store.spent_usd():.4f} of the ${study.budget_usd:.2f} budget")
        return 0

    try:
        topics = load_topics(study.path(study.topics_dir), study.topics)
    except TopicFileError as exc:
        print(f"Refused: {exc}")
        return 1
    version = code_version()
    rows = estimate(study, topics, registry, template)
    print(f"Study {study.study_id}: {len(topics)} topics x {len(study.endpoints)} endpoints x {study.runs} runs "
          f"= {len(topics) * len(study.endpoints) * study.runs} calls of {study.items_per_call} items")
    print(f"template {template.version} ({template.sha256[:12]}), code {version or 'unknown'}, database {database}")
    unverified = sorted(t.topic_id for t in topics
                        if (t.meta.get("provenance") or {}).get("verification", "unverified") == "unverified")
    if unverified:
        print(f"NOTE: {len(unverified)} topic texts are not verified yet: {', '.join(unverified)}")
    print_table(rows)
    print(f"rough total ${sum(r['est_cost_usd'] for r in rows):.2f} (budget ${study.budget_usd:.2f}); "
          f"it assumes about {EST_OUTPUT_TOKENS} output tokens per call")
    if args.plan:
        return 0
    if study.require_clean_code and (not version or version.endswith("+dirty")):
        print("Refused: this study must run from committed code (commit, tag, then start).")
        return 1
    if not args.yes and input("Start? [y/N] ").strip().lower() != "y":
        return 1

    store = Store(database)
    try:
        with run_lock(database):
            cells = prepare(study, store, topics, registry, template, version)
            runner = Runner(study, store, topics, cells, registry, template)
            runner.run()
    except (RunLockedError, StudyMismatchError) as exc:
        print(f"Refused: {exc}")
        return 1
    print()
    print_table(status_rows(store, study.max_attempts))
    print(f"spent: ${store.spent_usd():.4f} of the ${study.budget_usd:.2f} budget")
    for alias, reason in runner.stopped_endpoints.items():
        print(f"stopped {alias}: {reason}")
    if runner.stop_reason:
        print(f"run stopped: {runner.stop_reason}")
    pending = [c for c in cells if not store.is_done(c.cell_id)]
    print("all cells done" if not pending else f"{len(pending)} cells not done; run the same command to resume")
    return 0 if not pending else 2


if __name__ == "__main__":
    sys.exit(main())
