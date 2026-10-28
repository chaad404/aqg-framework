import pytest

from aqg.endpoints.protocols import EmptyResponseError, UnsupportedSettingError, build_body, parse_response
from aqg.endpoints.registry import load_registry

MESSAGES = [{"role": "system", "content": "Output ONLY JSON."}, {"role": "user", "content": "Write 5 MCQs."}]


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_openrouter_body_pins_the_host_and_adds_extras(registry):
    spec = registry.get("gpt-oss")
    body = build_body(spec, MESSAGES, spec.settings)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == MESSAGES
    assert body["provider"] == {"order": ["groq"], "allow_fallbacks": False, "require_parameters": True}
    assert body["reasoning"] == {"effort": "medium"}
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.7, 1.0, 8192)
    assert "response_format" not in body          # plain text: structure is measured, not forced


def test_mistral_spells_the_seed_its_own_way(registry):
    spec = registry.get("mistral-small")
    body = build_body(spec, MESSAGES, {**spec.settings, "seed": 7})
    assert body["random_seed"] == 7 and "seed" not in body
    assert "provider" not in body


def test_cohere_spells_top_p_as_p(registry):
    spec = registry.get("cohere")
    body = build_body(spec, MESSAGES, {**spec.settings, "seed": 7})
    assert body["p"] == 1.0 and "top_p" not in body
    assert body["seed"] == 7


def test_unknown_setting_is_refused(registry):
    with pytest.raises(UnsupportedSettingError, match="temprature"):
        build_body(registry.get("cohere"), MESSAGES, {"temprature": 0.2})


def test_parse_openrouter_response():
    data = {
        "model": "openai/gpt-oss-120b", "provider": "Groq",
        "choices": [{"message": {"role": "assistant", "content": "{\"questions\": []}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 900,
                  "completion_tokens_details": {"reasoning_tokens": 400}, "cost": 0.00072},
    }
    parsed = parse_response("openrouter", data)
    assert parsed.text == "{\"questions\": []}"
    assert (parsed.returned_model_id, parsed.served_by, parsed.finish_reason) == ("openai/gpt-oss-120b", "Groq", "stop")
    assert (parsed.prompt_tokens, parsed.completion_tokens, parsed.reasoning_tokens) == (1200, 900, 400)
    assert parsed.billed_cost_usd == pytest.approx(0.00072)


def test_parse_mistral_response_names_mistral_as_host():
    data = {"model": "mistral-small-2603", "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    parsed = parse_response("mistral", data)
    assert (parsed.served_by, parsed.reasoning_tokens, parsed.billed_cost_usd) == ("mistral", None, None)


def test_parse_cohere_response_joins_text_parts():
    data = {"message": {"role": "assistant", "content": [{"type": "text", "text": "{\"questions\": "},
                                                         {"type": "text", "text": "[]}"}]},
            "finish_reason": "COMPLETE", "usage": {"tokens": {"input_tokens": 50, "output_tokens": 20}}}
    parsed = parse_response("cohere", data)
    assert parsed.text == "{\"questions\": []}"
    assert (parsed.prompt_tokens, parsed.completion_tokens, parsed.returned_model_id) == (50, 20, None)


@pytest.mark.parametrize("provider,data", [
    ("openrouter", {"choices": []}),
    ("openrouter", {"error": {"message": "upstream overloaded"}}),
    ("cohere", {"message": {"content": []}}),
    ("openrouter", {"choices": ["oops"]}),
    ("openrouter", []),
    ("mistral", {"choices": [{"message": {"content": 42}}]}),
    ("cohere", {"message": "oops"}),
])
def test_response_without_a_usable_message_is_an_error(provider, data):
    # Unexpected shapes must become a retryable error, never crash a run
    with pytest.raises(EmptyResponseError):
        parse_response(provider, data)


def test_body_never_shares_objects_with_the_registry(registry):
    spec = registry.get("gpt-oss")
    body = build_body(spec, MESSAGES, spec.settings)
    body["provider"]["order"].append("deepinfra")
    body["reasoning"]["effort"] = "high"
    assert spec.routing["order"] == ["groq"] and spec.extra["reasoning"]["effort"] == "medium"


def test_content_given_as_parts_keeps_only_the_text():
    data = {"choices": [{"message": {"content": [{"type": "thinking", "thinking": "hmm"},
                                                 {"type": "text", "text": "OK"}]}, "finish_reason": "stop"}]}
    assert parse_response("mistral", data).text == "OK"


def test_cohere_reports_processed_and_billed_tokens():
    # Measured 28 Oct: Cohere adds a hidden preamble of ~495 tokens that it processes but does not bill
    data = {"message": {"content": [{"type": "text", "text": "OK"}]}, "finish_reason": "COMPLETE",
            "usage": {"billed_units": {"input_tokens": 7, "output_tokens": 2},
                      "tokens": {"input_tokens": 502, "output_tokens": 5}}}
    parsed = parse_response("cohere", data)
    assert (parsed.prompt_tokens, parsed.completion_tokens) == (502, 5)
    assert (parsed.billed_prompt_tokens, parsed.billed_completion_tokens) == (7, 2)
