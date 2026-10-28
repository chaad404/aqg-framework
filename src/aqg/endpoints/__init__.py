"""Model endpoints: the registry of pinned models and the HTTP client that calls them."""
from aqg.endpoints.client import Completion, EndpointClient, EndpointError, RateLimiter
from aqg.endpoints.registry import (
    THESIS_ENDPOINTS,
    EndpointSpec,
    Registry,
    UnknownEndpointError,
    get_registry,
    load_registry,
)

__all__ = [
    "THESIS_ENDPOINTS",
    "Completion",
    "EndpointClient",
    "EndpointError",
    "EndpointSpec",
    "RateLimiter",
    "Registry",
    "UnknownEndpointError",
    "get_registry",
    "load_registry",
]
