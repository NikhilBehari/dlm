"""Provider protocol and offline test backend."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_INDEX_PATTERN = re.compile(r"index\s*[:=]?\s*(\d+)", flags=re.IGNORECASE)

_RATE_LIMIT_SIGNALS = (
    "rate limit",
    "ratelimit",
    "rate_limit",
    "quota",
    "resource_exhausted",
    "too many requests",
    " 429",
    "(429)",
)


def _is_rate_limit(exc: BaseException) -> bool:
    """Heuristically detect rate-limit / quota errors across provider SDKs."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in _RATE_LIMIT_SIGNALS)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by every LLM backend.

    :meth:`pick_index` returns ``(chosen_index, raw_response)`` so that the
    full LLM output for the selection step is available to the caller (for
    logging and transcripts), not only the parsed integer.
    """

    def propose_reward(self, prompt: str) -> str: ...
    def pick_index(self, prompt: str, n_options: int) -> tuple[int, str]: ...


def _parse_index(text: str, n_options: int) -> int:
    match = _INDEX_PATTERN.search(text)
    if match is None:
        return 0
    return max(0, min(int(match.group(1)), n_options - 1))


class EchoProvider:
    """Cycles through preset responses; no network access.

    Args:
        rewards: expressions returned on successive :meth:`propose_reward` calls.
        picks: indices returned on successive :meth:`pick_index` calls; each
            value is clamped to ``[0, n_options)`` at call time.
    """

    def __init__(
        self,
        rewards: list[str] | None = None,
        picks: list[int] | None = None,
    ) -> None:
        self.rewards = list(rewards) if rewards else ["state"]
        self.picks = list(picks) if picks else [0]
        self._reward_idx = 0
        self._pick_idx = 0

    def propose_reward(self, prompt: str) -> str:
        out = self.rewards[self._reward_idx % len(self.rewards)]
        self._reward_idx += 1
        return f"$$${out}$$$"

    def pick_index(self, prompt: str, n_options: int) -> tuple[int, str]:
        out = self.picks[self._pick_idx % len(self.picks)]
        self._pick_idx += 1
        idx = max(0, min(out, n_options - 1))
        return idx, f"index: {idx}"


class RemoteProvider:
    """Base class for hosted-model adapters; subclasses implement :meth:`_complete`."""

    def propose_reward(self, prompt: str) -> str:
        return self._complete(prompt)

    def pick_index(self, prompt: str, n_options: int) -> tuple[int, str]:
        text = self._complete(prompt)
        return _parse_index(text, n_options), text

    def _complete(self, prompt: str) -> str:  # pragma: no cover - subclass hook
        raise NotImplementedError
