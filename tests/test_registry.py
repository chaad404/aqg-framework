import pytest
from pydantic import ValidationError

from aqg.endpoints.registry import THESIS_ENDPOINTS, EndpointSpec, UnknownEndpointError, load_registry

# The compared endpoints. An alias must never be re-pointed to another model or host.
EXPECTED = {
    "gpt-oss": ("openai/gpt-oss-120b", "openrouter", "groq"),
    "llama33": ("meta-llama/llama-3.3-70b-instruct", "openrouter", "coreweave"),
    "mistral-small": ("mistral-small-2603", "mistral", "mistral"),
    "cohere": ("command-a-03-2025", "cohere", "cohere"),
}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_thesis_aliases_keep_their_models_and_hosts(registry):
    assert {s.alias: (s.model_id, s.provider, s.host) for s in registry.thesis()} == EXPECTED


def test_every_thesis_endpoint_gets_the_same_settings(registry):
    settings = [spec.settings for spec in registry.thesis()]
    assert all(s == settings[0] for s in settings)
    assert settings[0] == registry.defaults


def test_thesis_endpoints_share_one_size_band(registry):
    assert all(70 <= spec.params_total_b <= 120 for spec in registry.thesis())


def test_unknown_alias_is_refused(registry):
    with pytest.raises(UnknownEndpointError, match="llama3"):
        registry.get("llama3")


def test_retired_alias_is_only_read_on_request(tmp_path):
    path = tmp_path / "endpoints.yaml"
    path.write_text(
        "pricing_date: '2026-10-19'\n"
        "endpoints:\n"
        "  old:\n"
        "    {model_id: m, provider: cohere, host: cohere, api_key_env: K, architecture: dense,\n"
        "     params_total_b: 1, params_active_b: 1, params_source: s, rpm: 1,\n"
        "     price_per_m: {input: 1, output: 1}, retired: true}\n")
    registry = load_registry(path)
    with pytest.raises(UnknownEndpointError, match="retired"):
        registry.get("old")
    assert registry.get("old", allow_retired=True).model_id == "m"


def test_typo_in_an_endpoint_is_refused(tmp_path):
    path = tmp_path / "endpoints.yaml"
    path.write_text(
        "pricing_date: '2026-10-19'\n"
        "endpoints:\n"
        "  x:\n"
        "    {model_id: m, provider: cohere, host: cohere, api_key_env: K, architecture: dense,\n"
        "     params_total_b: 1, params_active_b: 1, params_source: s, rpm: 1,\n"
        "     price_per_m: {input: 1, output: 1}, temprature: 0.2}\n")
    with pytest.raises(ValidationError, match="temprature"):
        load_registry(path)


def test_gateway_endpoint_must_pin_its_host():
    base = dict(alias="x", model_id="m", provider="openrouter", host="groq", api_key_env="K",
                architecture="moe", params_total_b=1, params_active_b=1, params_source="s", rpm=1,
                price_per_m={"input": 1, "output": 1})
    EndpointSpec(**base, routing={"order": ["groq"], "allow_fallbacks": False})
    with pytest.raises(ValidationError, match="pin exactly its host"):
        EndpointSpec(**base, routing={"order": ["groq"], "allow_fallbacks": True})
    with pytest.raises(ValidationError, match="pin exactly its host"):
        EndpointSpec(**base, routing={"order": ["groq", "deepinfra"], "allow_fallbacks": False})


def test_thesis_list_matches_the_registry(registry):
    assert set(THESIS_ENDPOINTS) <= set(registry.endpoints)


def direct_spec(**overrides):
    base = dict(alias="x", model_id="m", provider="cohere", host="cohere", api_key_env="K", architecture="dense",
                params_total_b=1, params_active_b=1, params_source="s", rpm=1, price_per_m={"input": 1, "output": 1})
    return EndpointSpec(**{**base, **overrides})


@pytest.mark.parametrize("extra", [{"temperature": 1.0}, {"p": 0.5}, {"random_seed": 1}, {"model": "other"}])
def test_extra_cannot_override_shared_settings(extra):
    # Otherwise one endpoint could quietly get a different temperature and the comparison would be unfair
    with pytest.raises(ValidationError, match="extra may not set"):
        direct_spec(extra=extra)


def test_provider_specific_extra_is_allowed():
    assert direct_spec(extra={"safety_mode": "OFF"}).extra == {"safety_mode": "OFF"}


def test_routing_only_applies_to_the_gateway():
    with pytest.raises(ValidationError, match="routing only applies"):
        direct_spec(routing={"order": ["cohere"]})
