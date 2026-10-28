import httpx

from aqg.endpoints.check import check, main
from aqg.endpoints.registry import load_registry

REGISTRY = load_registry()


def http_returning(payload, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload)))


def test_gateway_endpoint_is_found_on_its_pinned_host():
    payload = {"data": {"endpoints": [{"tag": "deepinfra", "quantization": "fp8"},
                                      {"tag": "groq", "quantization": "unknown"}]}}
    ok, message = check(REGISTRY.get("gpt-oss"), http_returning(payload), "key")
    assert ok and "groq" in message


def test_wrong_precision_counts_as_missing():
    payload = {"data": {"endpoints": [{"tag": "coreweave", "quantization": "fp8"}]}}
    ok, message = check(REGISTRY.get("llama33"), http_returning(payload), "key")
    assert not ok and "fp16" in message


def test_listed_model_is_available():
    payload = {"models": [{"name": "command-a-03-2025"}, {"name": "command-r"}]}
    assert check(REGISTRY.get("cohere"), http_returning(payload), "key")[0]


def test_missing_key_fails_without_a_request():
    def refuse(request):
        raise AssertionError("no request expected")
    ok, message = check(REGISTRY.get("mistral-small"), httpx.Client(transport=httpx.MockTransport(refuse)), None)
    assert not ok and "MISTRAL_API_KEY" in message


def test_unknown_alias_is_reported_not_raised(capsys):
    assert main(["llama3"]) == 1
    assert "[UNKNOWN] llama3" in capsys.readouterr().out


def test_http_error_is_reported_not_raised():
    ok, message = check(REGISTRY.get("cohere"), http_returning({"message": "no"}, status=401), "key")
    assert not ok and "401" in message
