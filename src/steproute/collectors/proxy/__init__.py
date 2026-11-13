"""OpenAI-compatible proxy collector (issue 12). See ``docs/PROXY.md``.

Importing this package does not import ``litellm``; the first upstream call does (through
``upstream.load_litellm``, which switches LiteLLM's own network fetches off first).
"""

from steproute.collectors.proxy.config import (
    ProxyConfig,
    ProxyConfigError,
    UpstreamConfig,
    load_proxy_config,
)
from steproute.collectors.proxy.engine import SHADOW_PREFIX, Proxy, Response
from steproute.collectors.proxy.server import build_proxy, make_proxy_server, run_proxy
from steproute.collectors.proxy.source import (
    DecisionSource,
    HttpSource,
    InProcessSource,
    SourceError,
)

__all__ = [
    "SHADOW_PREFIX",
    "DecisionSource",
    "HttpSource",
    "InProcessSource",
    "Proxy",
    "ProxyConfig",
    "ProxyConfigError",
    "Response",
    "SourceError",
    "UpstreamConfig",
    "build_proxy",
    "load_proxy_config",
    "make_proxy_server",
    "run_proxy",
]
