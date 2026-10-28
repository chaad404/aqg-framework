"""Checks that the pinned endpoints are offered right now. Run it before every pilot or run.

Listing models is free; nothing is generated.

    python -m aqg.endpoints.check              # the four thesis endpoints
    python -m aqg.endpoints.check gpt-oss      # selected aliases

Exit code 0 = everything available, 1 = something is missing: do not start the run.
"""
import sys

import httpx

from aqg import config
from aqg.endpoints.registry import THESIS_ENDPOINTS, EndpointSpec, UnknownEndpointError, get_registry

OPENROUTER_ENDPOINTS = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
MODEL_LISTS = {   # provider: (url, list field, id field)
    "mistral": ("https://api.mistral.ai/v1/models", "data", "id"),
    "cohere": ("https://api.cohere.com/v1/models?page_size=1000", "models", "name"),
}


def check(spec: EndpointSpec, http: httpx.Client, api_key: str | None) -> tuple[bool, str]:
    """(available, message) for one endpoint. Messages never contain the key."""
    if not api_key:
        return False, f"{spec.api_key_env} is not set: add it to the secrets file"
    try:
        if spec.provider == "openrouter":
            response = http.get(OPENROUTER_ENDPOINTS.format(model_id=spec.model_id))
            response.raise_for_status()
            offered = (response.json().get("data") or {}).get("endpoints") or []
            quantizations = spec.routing.get("quantizations")
            serving = [e for e in offered if (e.get("tag") or "").split("/")[0] == spec.host
                       and (not quantizations or e.get("quantization") in quantizations)]
            if not serving:
                available = sorted(f"{e.get('tag')} ({e.get('quantization')})" for e in offered)
                return False, f"no {spec.host} endpoint with precision {quantizations or 'any'}; offered: {available}"
            return True, f"{spec.model_id} served by {serving[0].get('tag')} ({serving[0].get('quantization')})"
        url, list_field, id_field = MODEL_LISTS[spec.provider]
        response = http.get(url, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        listed = {model.get(id_field) for model in response.json().get(list_field) or []}
        if spec.model_id not in listed:
            return False, f"{spec.model_id} is not listed by {spec.provider}"
        return True, f"{spec.model_id} listed by {spec.provider}"
    except (httpx.HTTPError, ValueError) as exc:
        detail = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
        return False, f"request to {spec.provider} failed: {detail}"


def main(aliases: list[str]) -> int:
    registry = get_registry()
    all_ok = True
    with httpx.Client(timeout=30) as http:
        for alias in aliases:
            try:
                spec = registry.get(alias)
            except UnknownEndpointError as exc:
                all_ok = False
                print(f"[UNKNOWN] {alias}: {exc}")
                continue
            ok, message = check(spec, http, config.get_secret(spec.api_key_env))
            all_ok &= ok
            print(f"[{'OK' if ok else 'MISSING'}] {alias}: {message}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(THESIS_ENDPOINTS)))
