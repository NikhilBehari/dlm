"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import os

from .base import RemoteProvider


class OpenAIProvider(RemoteProvider):
    """OpenAI Chat Completions adapter (``openai`` SDK)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("OpenAIProvider requires `pip install openai`.") from exc
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def _complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""
