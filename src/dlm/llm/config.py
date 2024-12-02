"""Load an :class:`LLMProvider` from a TOML config file."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def load_provider(path: str | Path = "llm_config.toml") -> LLMProvider:
    """Build a provider from a TOML file.

    The file must declare ``provider`` (``openai`` / ``anthropic`` /
    ``gemini``) and ``api_key``. Remaining keys are forwarded to the
    provider constructor (``model``, ``temperature``, ``max_tokens``, ...).
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LLM config not found at {cfg_path}")
    with cfg_path.open("rb") as f:
        config = tomllib.load(f)
    try:
        name = config.pop("provider")
    except KeyError as exc:
        raise KeyError(f"missing required `provider` field in {cfg_path}") from exc
    if name not in _PROVIDERS:
        raise ValueError(
            f"unknown provider {name!r} in {cfg_path}; "
            f"expected one of {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name](**config)
