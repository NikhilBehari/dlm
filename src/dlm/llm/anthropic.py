"""Anthropic Messages API adapter."""

from __future__ import annotations

import os

from .base import RemoteProvider


class AnthropicProvider(RemoteProvider):
    """Anthropic Messages API adapter (``anthropic`` SDK)."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("AnthropicProvider requires `pip install anthropic`.") from exc
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _complete(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in resp.content)
