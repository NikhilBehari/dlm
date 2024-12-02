"""Google Gen AI (Gemini) adapter."""

from __future__ import annotations

import os

from .base import RemoteProvider


class GeminiProvider(RemoteProvider):
    """Google Gen AI adapter (``google-genai`` SDK).

    The API key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
    if not passed explicitly.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("GeminiProvider requires `pip install google-genai`.") from exc
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature

    def _complete(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        return resp.text or ""
