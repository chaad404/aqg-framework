import json

import httpx
import pytest

from aqg.endpoints.client import MAX_ATTEMPTS, EndpointClient, EndpointError, RateLimiter
from aqg.endpoints.registry import load_registry

FAKE_KEY = "test-key-0123456789abcdef"
MESSAGES = [{"role": "user", "content": "Write 5 MCQs."}]
OK_BODY = {"model": "openai/gpt-oss-120b", "provider": "Groq",
           "choices": [{"message": {"content": "{\"questions\": []}"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}


class Script:
    """Answers each request with the next scripted response and records what was sent."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request):
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make_client(alias, script, sleeps):
    spec = load_registry().get(alias)
    http = httpx.Client(transport=httpx.MockTransport(script))
    return EndpointClient(spec, api_key=FAKE_KEY, http=http, limiter=RateLimiter(None),
                          sleep=sleeps.append, jitter=lambda: 0.0)


def test_successful_call_records_provenance():
    script, sleeps = Script(httpx.Response(200, json=OK_BODY)), []
    completion = make_client("gpt-oss", script, sleeps).complete(MESSAGES)
    sent = script.requests[0]
    assert str(sent.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(sent.content)["provider"]["order"] == ["groq"]
    assert completion.attempts == 1 and completion.retry_log == [] and sleeps == []
    assert (completion.requested_model_id, completion.returned_model_id, completion.served_by) == (
        "openai/gpt-oss-120b", "openai/gpt-oss-120b", "Groq")
    assert completion.estimated_cost_usd == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)
    assert completion.request_body["messages"] == MESSAGES and completion.response_body == OK_BODY
    assert completion.latency_s >= 0 and completion.elapsed_s >= completion.latency_s


def test_rate_limit_waits_as_told_then_succeeds():
    script, sleeps = Script(httpx.Response(429, headers={"Retry-After": "3"}, text="slow down"),
                            httpx.Response(200, json=OK_BODY)), []
    completion = make_client("gpt-oss", script, sleeps).complete(MESSAGES)
    assert completion.attempts == 2 and sleeps == [3.0]
    assert completion.retry_log == [{"attempt": 1, "status": 429, "error": "HTTP 429: slow down", "wait_s": 3.0}]


def test_network_error_and_empty_answer_are_retried():
    script, sleeps = Script(httpx.ConnectError("refused"), httpx.Response(200, json={"choices": []}),
                            httpx.Response(200, json=OK_BODY)), []
    completion = make_client("gpt-oss", script, sleeps).complete(MESSAGES)
    assert completion.attempts == 3 and sleeps == [2.0, 4.0]          # exponential backoff, no jitter
    assert [entry["status"] for entry in completion.retry_log] == [None, 200]


def test_gives_up_after_the_last_attempt_and_keeps_every_attempt():
    script, sleeps = Script(*[httpx.Response(503, text="busy")] * MAX_ATTEMPTS), []
    with pytest.raises(EndpointError) as err:
        make_client("cohere", script, sleeps).complete(MESSAGES)
    assert err.value.attempts == MAX_ATTEMPTS and len(err.value.retry_log) == MAX_ATTEMPTS
    assert len(sleeps) == MAX_ATTEMPTS - 1 and err.value.status == 503


def test_auth_error_stops_at_once_and_never_shows_the_key():
    script, sleeps = Script(httpx.Response(401, text=f"invalid key {FAKE_KEY}")), []
    with pytest.raises(EndpointError) as err:
        make_client("mistral-small", script, sleeps).complete(MESSAGES)
    assert err.value.attempts == 1 and sleeps == []
    assert FAKE_KEY not in str(err.value) and FAKE_KEY not in json.dumps(err.value.retry_log)


def test_requests_go_to_each_providers_own_api():
    for alias, url in [("mistral-small", "https://api.mistral.ai/v1/chat/completions"),
                       ("cohere", "https://api.cohere.com/v2/chat")]:
        body = OK_BODY if alias != "cohere" else {"message": {"content": [{"type": "text", "text": "x"}]},
                                                  "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}}}
        script = Script(httpx.Response(200, json=body))
        make_client(alias, script, []).complete(MESSAGES)
        assert str(script.requests[0].url) == url


def test_zero_quota_fails_at_once_with_a_clear_message():
    # Seen 28 Oct: a Mistral key still on free mode answers 429 with a limit of 0 requests a minute
    script = Script(httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0"}, text="Rate limit exceeded"))
    sleeps = []
    with pytest.raises(EndpointError) as err:
        make_client("mistral-small", script, sleeps).complete(MESSAGES)
    assert err.value.attempts == 1 and sleeps == [] and "activate paid use" in str(err.value)


def test_cohere_cost_is_estimated_from_billed_tokens():
    body = {"message": {"content": [{"type": "text", "text": "OK"}]}, "finish_reason": "COMPLETE",
            "usage": {"billed_units": {"input_tokens": 7, "output_tokens": 2},
                      "tokens": {"input_tokens": 502, "output_tokens": 5}}}
    completion = make_client("cohere", Script(httpx.Response(200, json=body)), []).complete(MESSAGES)
    assert (completion.prompt_tokens, completion.billed_prompt_tokens) == (502, 7)
    assert completion.estimated_cost_usd == pytest.approx((7 * 2.50 + 2 * 10.00) / 1e6)


def test_rate_limiter_spaces_out_calls():
    # Three calls at the same instant: the second waits one slot (1 s at 60 rpm), the third two slots.
    limiter, waits = RateLimiter(rpm=60), []
    for _ in range(3):
        limiter.wait(sleep=waits.append)
    assert len(waits) == 2
    assert waits[0] == pytest.approx(1.0, abs=0.05) and waits[1] == pytest.approx(2.0, abs=0.05)
