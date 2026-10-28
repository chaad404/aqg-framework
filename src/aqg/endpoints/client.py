"""Calls one endpoint: rate limiting, retries with backoff and a full provenance record for every call.

A call either returns a Completion or raises EndpointError; both carry every attempt, so failures are
counted in the results, never hidden.
"""
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import httpx

from aqg import config
from aqg.endpoints.protocols import API_URLS, build_body, parse_response
from aqg.endpoints.registry import EndpointSpec

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4            # the first try plus up to 3 retries, as in the experiment protocol
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0
MAX_RETRY_AFTER_S = 120.0
TIMEOUT_S = 180.0


class EndpointError(RuntimeError):
    """A call that failed for good, with its attempts."""

    def __init__(self, message: str, alias: str, status: int | None, attempts: int, retry_log: list):
        super().__init__(message)
        self.alias, self.status, self.attempts, self.retry_log = alias, status, attempts, retry_log


@dataclass
class Completion:
    alias: str
    provider: str
    host: str
    requested_model_id: str
    returned_model_id: str | None
    served_by: str | None
    text: str
    finish_reason: str | None
    prompt_tokens: int | None            # tokens the model processed
    completion_tokens: int | None
    reasoning_tokens: int | None
    billed_prompt_tokens: int | None     # tokens billed, when the provider reports them separately (Cohere)
    billed_completion_tokens: int | None
    billed_cost_usd: float | None        # cost reported by the provider
    estimated_cost_usd: float | None     # billed (or else processed) tokens x registry prices
    latency_s: float                 # the successful request only: the endpoint's own latency
    elapsed_s: float                 # the whole call, including rate-limit waits, failed attempts and backoff
    attempts: int
    started_at: str
    settings: dict = field(default_factory=dict)
    retry_log: list = field(default_factory=list)
    request_body: dict = field(default_factory=dict)
    response_body: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RateLimiter:
    """Spaces out requests so an endpoint never gets more than `rpm` calls a minute, across threads."""

    def __init__(self, rpm: float | None):
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self, sleep=time.sleep) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        if slot > now:
            sleep(slot - now)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(spec: EndpointSpec) -> RateLimiter:
    """One limiter per endpoint, shared by every client and worker thread in the process."""
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault(spec.alias, RateLimiter(spec.rpm))


class EndpointClient:
    def __init__(self, spec: EndpointSpec, api_key: str | None = None, http: httpx.Client | None = None,
                 limiter: RateLimiter | None = None, sleep=time.sleep, clock=time.perf_counter, jitter=random.random):
        self.spec = spec
        self._api_key = api_key or config.require_secret(spec.api_key_env)
        self._http = http or httpx.Client(timeout=TIMEOUT_S)
        self._limiter = limiter or shared_limiter(spec)
        self._sleep, self._clock, self._jitter = sleep, clock, jitter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._http.close()

    def complete(self, messages: list[dict], overrides: dict | None = None) -> Completion:
        settings = {**self.spec.settings, **(overrides or {})}
        body = build_body(self.spec, messages, settings)
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start, retry_log = self._clock(), []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._limiter.wait(self._sleep)
            sent = self._clock()
            status, retry_after, no_quota = None, None, False
            try:
                response = self._http.post(API_URLS[self.spec.provider], json=body, headers=self._headers())
                status = response.status_code
                if status == 200:
                    data = response.json()
                    parsed = parse_response(self.spec.provider, data)
                    return self._completion(parsed, data, body, settings, latency=self._clock() - sent,
                                            elapsed=self._clock() - start, attempts=attempt,
                                            retry_log=retry_log, started_at=started_at)
                error = f"HTTP {status}: {response.text[:300]}"
                retry_after = _retry_after(response)
                if status == 429 and response.headers.get("x-ratelimit-limit-req-minute") == "0":
                    no_quota = True         # a free plan without this model: waiting cannot help
                    error += " (the account allows 0 requests a minute for this model: activate paid use)"
            except httpx.RequestError as exc:  # timeouts, network and decoding errors
                error = f"{type(exc).__name__}: {exc}"
            except ValueError as exc:          # invalid JSON or a response without a message
                error = f"{type(exc).__name__}: {exc}"
            retryable = not no_quota and (status is None or status == 200 or status in RETRY_STATUSES)
            wait = self._backoff(attempt, retry_after) if retryable and attempt < MAX_ATTEMPTS else 0.0
            retry_log.append({"attempt": attempt, "status": status, "error": self._redact(error), "wait_s": round(wait, 2)})
            if not retryable:
                raise EndpointError(f"{self.spec.alias}: {self._redact(error)}", self.spec.alias, status, attempt, retry_log)
            if attempt < MAX_ATTEMPTS:
                self._sleep(wait)
        raise EndpointError(f"{self.spec.alias}: failed after {MAX_ATTEMPTS} attempts; last: {retry_log[-1]['error']}",
                            self.spec.alias, retry_log[-1]["status"], MAX_ATTEMPTS, retry_log)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, MAX_RETRY_AFTER_S)
        return min(BASE_BACKOFF_S * 2 ** (attempt - 1), MAX_BACKOFF_S) + self._jitter()

    def _completion(self, parsed, data, body, settings, latency, elapsed, attempts, retry_log, started_at):
        prices = self.spec.price_per_m
        tokens_in = parsed.billed_prompt_tokens if parsed.billed_prompt_tokens is not None else parsed.prompt_tokens
        tokens_out = (parsed.billed_completion_tokens if parsed.billed_completion_tokens is not None
                      else parsed.completion_tokens)
        estimated = None
        if tokens_in is not None and tokens_out is not None:
            estimated = (tokens_in * prices.input + tokens_out * prices.output) / 1e6
        return Completion(
            alias=self.spec.alias, provider=self.spec.provider, host=self.spec.host,
            requested_model_id=self.spec.model_id, returned_model_id=parsed.returned_model_id,
            served_by=parsed.served_by, text=parsed.text, finish_reason=parsed.finish_reason,
            prompt_tokens=parsed.prompt_tokens, completion_tokens=parsed.completion_tokens,
            reasoning_tokens=parsed.reasoning_tokens, billed_prompt_tokens=parsed.billed_prompt_tokens,
            billed_completion_tokens=parsed.billed_completion_tokens, billed_cost_usd=parsed.billed_cost_usd,
            estimated_cost_usd=estimated, latency_s=round(latency, 3), elapsed_s=round(elapsed, 3),
            attempts=attempts, started_at=started_at, settings=settings, retry_log=retry_log,
            request_body=body, response_body=data,
        )


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None                         # missing, or given as an HTTP date: use the normal backoff
