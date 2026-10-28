"""How each provider's chat API spells a request and a response.

Settings use one vocabulary (temperature, top_p, max_tokens, seed) and are translated here, so the same
settings mean the same thing for every endpoint. Requests ask for plain text: no provider JSON mode, so
the structural checks measure the endpoint itself.
"""
import copy
from dataclasses import dataclass

API_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "cohere": "https://api.cohere.com/v2/chat",
}

# The shared setting name -> the provider's parameter name
SETTING_NAMES = {
    "openrouter": {"temperature": "temperature", "top_p": "top_p", "max_tokens": "max_tokens", "seed": "seed"},
    "mistral": {"temperature": "temperature", "top_p": "top_p", "max_tokens": "max_tokens", "seed": "random_seed"},
    "cohere": {"temperature": "temperature", "top_p": "p", "max_tokens": "max_tokens", "seed": "seed"},
}

# Body keys an endpoint's `extra` may never set: they belong to the shared request or the shared settings
RESERVED_BODY_KEYS = {"model", "messages", "provider"} | {name for names in SETTING_NAMES.values()
                                                          for name in names.values()}


class UnsupportedSettingError(ValueError):
    pass


class EmptyResponseError(ValueError):
    """A successful HTTP status without a usable message; the client counts it as a failed attempt."""


@dataclass
class ParsedResponse:
    text: str
    finish_reason: str | None
    returned_model_id: str | None
    served_by: str | None
    prompt_tokens: int | None            # tokens the model processed
    completion_tokens: int | None
    reasoning_tokens: int | None
    billed_prompt_tokens: int | None     # tokens the provider bills, when it reports them separately (Cohere)
    billed_completion_tokens: int | None
    billed_cost_usd: float | None


def build_body(spec, messages: list[dict], settings: dict) -> dict:
    names = SETTING_NAMES[spec.provider]
    unknown = sorted(set(settings) - set(names))
    if unknown:
        raise UnsupportedSettingError(f"{spec.alias}: unknown settings {unknown}; allowed: {sorted(names)}")
    body = {"model": spec.model_id, "messages": copy.deepcopy(messages)}
    body.update({names[key]: value for key, value in settings.items() if value is not None})
    if spec.provider == "openrouter":
        body["provider"] = copy.deepcopy(spec.routing)
    body.update(copy.deepcopy(spec.extra))      # copies, so a stored request can never change the registry
    return body


def parse_response(provider: str, data) -> ParsedResponse:
    """Reads one provider response. Any unexpected shape becomes EmptyResponseError, so it is retried."""
    try:
        return _parse_cohere(data) if provider == "cohere" else _parse_chat_completions(provider, data)
    except EmptyResponseError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise EmptyResponseError(f"{provider}: unexpected response shape ({type(exc).__name__}: {exc})") from exc


def _parse_chat_completions(provider: str, data: dict) -> ParsedResponse:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        reason = f" ({data['error']})" if data.get("error") else ""
        raise EmptyResponseError(f"{provider}: the response has no choices{reason}")
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return ParsedResponse(
        text=_text_of(choices[0]["message"].get("content")),
        finish_reason=choices[0].get("finish_reason"),
        returned_model_id=data.get("model"),
        served_by=data.get("provider") or provider,   # OpenRouter names the host that answered
        prompt_tokens=_int(usage.get("prompt_tokens")),
        completion_tokens=_int(usage.get("completion_tokens")),
        reasoning_tokens=_int(details.get("reasoning_tokens")),
        billed_prompt_tokens=None,
        billed_completion_tokens=None,
        billed_cost_usd=_float(usage.get("cost")),
    )


def _parse_cohere(data: dict) -> ParsedResponse:
    message = data.get("message") or {}
    if not message.get("content"):
        raise EmptyResponseError("cohere: the response has no message content")
    usage = data.get("usage") or {}
    tokens, billed = usage.get("tokens") or {}, usage.get("billed_units") or {}
    return ParsedResponse(
        text=_text_of(message["content"]),
        finish_reason=data.get("finish_reason"),
        returned_model_id=None,          # Cohere does not echo the model; the requested ID is pinned
        served_by="cohere",
        prompt_tokens=_int(tokens.get("input_tokens")),       # includes Cohere's own hidden preamble
        completion_tokens=_int(tokens.get("output_tokens")),
        reasoning_tokens=None,
        billed_prompt_tokens=_int(billed.get("input_tokens")),
        billed_completion_tokens=_int(billed.get("output_tokens")),
        billed_cost_usd=None,
    )


def _text_of(content) -> str:
    """Message content as text: a plain string, or the text parts of a list (thinking parts are left out)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text") or "" for part in content
                       if isinstance(part, dict) and part.get("type", "text") == "text")
    raise TypeError(f"message content is a {type(content).__name__}")


def _int(value) -> int | None:
    return int(value) if value is not None else None


def _float(value) -> float | None:
    return float(value) if value is not None else None
