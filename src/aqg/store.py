"""One SQLite file per study: the settings, the topics as sent, the planned cells, every call attempt and every item.

Call and item rows can only be added: triggers refuse UPDATE and DELETE, so a failed attempt can never be
overwritten by a later success and every failure stays countable. A call and its items are written in one
transaction. The settings, the template and each topic text are fixed when a study starts; resuming with
anything different is refused, which makes the frozen protocol enforceable.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS study (
    study_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL,
    template_version TEXT NOT NULL, template_sha256 TEXT NOT NULL, endpoints_json TEXT NOT NULL,
    pricing_date TEXT, code_version TEXT
);
CREATE TABLE IF NOT EXISTS topic (
    topic_id TEXT PRIMARY KEY, domain TEXT NOT NULL, title TEXT NOT NULL, text_sha256 TEXT NOT NULL,
    text TEXT NOT NULL, meta_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cell (
    cell_id TEXT PRIMARY KEY, topic_id TEXT NOT NULL REFERENCES topic, endpoint TEXT NOT NULL,
    run_no INTEGER NOT NULL, order_no INTEGER NOT NULL, UNIQUE (topic_id, endpoint, run_no)
);
CREATE TABLE IF NOT EXISTS call (
    call_id INTEGER PRIMARY KEY, cell_id TEXT NOT NULL REFERENCES cell, attempt INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected', 'error')),
    started_at TEXT NOT NULL, finished_at TEXT NOT NULL, endpoint TEXT NOT NULL,
    provider TEXT, host TEXT, requested_model_id TEXT, returned_model_id TEXT, served_by TEXT,
    settings_json TEXT, seed INTEGER, prompt_sha256 TEXT NOT NULL,
    http_attempts INTEGER, retry_log_json TEXT, latency_s REAL, elapsed_s REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
    billed_prompt_tokens INTEGER, billed_completion_tokens INTEGER, billed_cost_usd REAL, estimated_cost_usd REAL,
    finish_reason TEXT, reply_text TEXT,
    json_strict INTEGER, parse_mode TEXT, reply_strict INTEGER, repairs_json TEXT, errors_json TEXT,
    request_json TEXT, response_json TEXT,
    UNIQUE (cell_id, attempt)
);
CREATE TABLE IF NOT EXISTS item (
    item_id INTEGER PRIMARY KEY, call_id INTEGER NOT NULL REFERENCES call, cell_id TEXT NOT NULL REFERENCES cell,
    position INTEGER NOT NULL, question TEXT NOT NULL,
    option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL, option_d TEXT NOT NULL,
    answer TEXT NOT NULL, raw_answer TEXT NOT NULL, permutation TEXT NOT NULL, evidence TEXT NOT NULL,
    strict_ok INTEGER NOT NULL, repairs_json TEXT NOT NULL, flags_json TEXT NOT NULL, evidence_score REAL NOT NULL,
    UNIQUE (call_id, position)
);
"""
APPEND_ONLY = "".join(
    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} ON {table} "
    f"BEGIN SELECT RAISE(ABORT, '{table} rows are append-only'); END;\n"
    for table in ("call", "item") for action in ("UPDATE", "DELETE"))

CALL_COLUMNS = (
    "cell_id", "attempt", "status", "started_at", "finished_at", "endpoint", "provider", "host",
    "requested_model_id", "returned_model_id", "served_by", "settings_json", "seed", "prompt_sha256",
    "http_attempts", "retry_log_json", "latency_s", "elapsed_s", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "billed_prompt_tokens", "billed_completion_tokens", "billed_cost_usd", "estimated_cost_usd",
    "finish_reason", "reply_text", "json_strict", "parse_mode", "reply_strict", "repairs_json", "errors_json",
    "request_json", "response_json",
)
ITEM_COLUMNS = (
    "call_id", "cell_id", "position", "question", "option_a", "option_b", "option_c", "option_d", "answer",
    "raw_answer", "permutation", "evidence", "strict_ok", "repairs_json", "flags_json", "evidence_score",
)


class StudyMismatchError(RuntimeError):
    """The database was started with other settings, another template or other topic texts."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Store:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()          # one writer at a time; worker threads share the connection
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(SCHEMA + APPEND_ONLY)

    def close(self) -> None:
        self._db.close()

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # --- setting up or resuming a study ---------------------------------------------------------------

    def open_study(self, study_id: str, config: dict, config_sha256: str, template, endpoints: dict,
                   pricing_date: str, code_version: str | None) -> bool:
        """Records the study, or checks that a resumed one is the same. Returns True when it is new."""
        with self._lock:
            row = self._db.execute("SELECT * FROM study").fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO study VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (study_id, now(), dumps(config), config_sha256, template.version, template.sha256,
                     dumps(endpoints), pricing_date, code_version))
                return True
        problems = []
        if row["study_id"] != study_id:
            problems.append(f"it holds study '{row['study_id']}', not '{study_id}'")
        if row["config_sha256"] != config_sha256:
            problems.append("the study settings changed")
        if row["template_sha256"] != template.sha256:
            problems.append(f"the prompt template changed (was {row['template_version']})")
        if json.loads(row["endpoints_json"]) != endpoints:
            problems.append("an endpoint definition changed")
        if problems:
            raise StudyMismatchError("This database cannot be resumed: " + "; ".join(problems)
                                     + ". Start a new study with a new study_id and database.")
        return False

    def add_topics(self, topics) -> None:
        with self._lock:
            stored = {r["topic_id"]: r["text_sha256"] for r in self._db.execute("SELECT * FROM topic")}
            changed = [t.topic_id for t in topics if t.topic_id in stored and stored[t.topic_id] != t.sha256]
            if changed:
                raise StudyMismatchError(f"The text of {changed} changed since this study started; the topic texts "
                                         "are frozen for a study. Start a new study to use the new texts.")
            self._db.executemany(
                "INSERT OR IGNORE INTO topic VALUES (?, ?, ?, ?, ?, ?)",
                [(t.topic_id, t.domain, t.title, t.sha256, t.text, dumps(t.meta)) for t in topics])

    def add_cells(self, cells) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO cell VALUES (?, ?, ?, ?, ?)",
                [(c.cell_id, c.topic_id, c.endpoint, c.run_no, c.order_no) for c in cells])

    # --- progress --------------------------------------------------------------------------------------

    def attempts(self, cell_id: str) -> int:
        return self.query("SELECT COUNT(*) FROM call WHERE cell_id = ?", (cell_id,))[0][0]

    def is_done(self, cell_id: str) -> bool:
        return bool(self.query("SELECT 1 FROM call WHERE cell_id = ? AND status = 'accepted'", (cell_id,)))

    def spent_usd(self) -> float:
        """Billed cost where the provider reports it, otherwise the estimate."""
        return self.query("SELECT COALESCE(SUM(COALESCE(billed_cost_usd, estimated_cost_usd)), 0) FROM call")[0][0]

    # --- recording -------------------------------------------------------------------------------------

    def record_call(self, call: dict, items: list[dict] = ()) -> int:
        unknown = set(call) - set(CALL_COLUMNS)
        if unknown:
            raise KeyError(f"unknown call fields {sorted(unknown)}")
        with self._lock:
            try:
                self._db.execute("BEGIN")
                cursor = self._db.execute(
                    f"INSERT INTO call ({', '.join(CALL_COLUMNS)}) VALUES ({', '.join('?' * len(CALL_COLUMNS))})",
                    [call.get(column) for column in CALL_COLUMNS])
                call_id = cursor.lastrowid
                for item in items:
                    row = {**item, "call_id": call_id, "cell_id": call["cell_id"]}
                    self._db.execute(
                        f"INSERT INTO item ({', '.join(ITEM_COLUMNS)}) VALUES ({', '.join('?' * len(ITEM_COLUMNS))})",
                        [row[column] for column in ITEM_COLUMNS])
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return call_id
