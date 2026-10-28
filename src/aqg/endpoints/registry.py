"""The endpoint registry: configs/endpoints.yaml is the single source of truth for what each alias means.

An alias is never re-pointed. Environment variables cannot change an endpoint, and unknown keys in the
file are refused, so a typo cannot silently drop a setting.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aqg.config import REPO_ROOT
from aqg.endpoints.protocols import RESERVED_BODY_KEYS

REGISTRY_PATH = REPO_ROOT / "configs" / "endpoints.yaml"
THESIS_ENDPOINTS = ("gpt-oss", "llama33", "mistral-small", "cohere")


class UnknownEndpointError(ValueError):
    pass


class Prices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class EndpointSpec(BaseModel):
    """One pinned endpoint: model, host, settings and the facts reported about it in the thesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str
    model_id: str = Field(min_length=1)
    provider: Literal["openrouter", "mistral", "cohere"]
    host: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    routing: dict = Field(default_factory=dict)
    extra: dict = Field(default_factory=dict)
    settings: dict = Field(default_factory=dict)
    architecture: Literal["dense", "moe"]
    params_total_b: float = Field(gt=0)
    params_active_b: float = Field(gt=0)
    params_source: str
    rpm: float = Field(gt=0)
    price_per_m: Prices
    retired: bool = False

    @model_validator(mode="after")
    def _gateway_pins_one_host(self):
        if self.provider == "openrouter":
            if self.routing.get("order") != [self.host] or self.routing.get("allow_fallbacks", True):
                raise ValueError(f"{self.alias}: an OpenRouter endpoint must pin exactly its host "
                                 f"({self.host}) and set allow_fallbacks: false")
        elif self.routing:
            raise ValueError(f"{self.alias}: routing only applies to OpenRouter endpoints")
        return self

    @model_validator(mode="after")
    def _extra_cannot_override_shared_settings(self):
        clashes = sorted(set(self.extra) & RESERVED_BODY_KEYS)
        if clashes:
            raise ValueError(f"{self.alias}: extra may not set {clashes}; shared settings belong in `defaults`, "
                             "so every endpoint gets the same ones")
        return self


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pricing_date: str
    defaults: dict = Field(default_factory=dict)
    endpoints: dict[str, EndpointSpec]

    def get(self, alias: str, allow_retired: bool = False) -> EndpointSpec:
        if alias not in self.endpoints:
            raise UnknownEndpointError(f"Unknown endpoint '{alias}'. Known: {', '.join(self.endpoints)}")
        spec = self.endpoints[alias]
        if spec.retired and not allow_retired:
            raise UnknownEndpointError(f"Endpoint '{alias}' is retired; it stays only to read old runs")
        return spec

    def thesis(self) -> list[EndpointSpec]:
        return [self.get(alias) for alias in THESIS_ENDPOINTS]


def load_registry(path: Path | str = REGISTRY_PATH) -> Registry:
    """Reads the registry file; every endpoint gets the shared defaults, then its own settings on top."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    endpoints = {
        alias: {**spec, "alias": alias, "settings": {**defaults, **(spec.get("settings") or {})}}
        for alias, spec in (raw.get("endpoints") or {}).items()
    }
    return Registry(pricing_date=str(raw.get("pricing_date", "")), defaults=defaults, endpoints=endpoints)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """The repository's registry, read once per process."""
    return load_registry()
