"""Logging and result persistence for DLM runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def default_output_dir(task_name: str, root: str | Path = "outputs") -> Path:
    """Return ``outputs/<YYYY-MM-DD-HHMMSS-task-name>/`` (creation deferred)."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9]+", "-", task_name).strip("-")
    return Path(root) / f"{timestamp}-{safe}"


class OutputWriter:
    """Persist a DLM run to ``output_dir``.

    Files written:

      config.json       task and hyperparameter snapshot.
      result.json       final summary: best expression and per-stage breakdown.
      run.log           plain-text progress log.
      transcript.md     readable narrative interleaving prompts, responses, and evaluations.
      training.jsonl    one JSON line per training epoch, tagged with stage + candidate.
      prompts.jsonl     one JSON line per LLM interaction (prompt + raw response).
      candidates.jsonl  one JSON line per evaluated candidate.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.dir / "run.log").open("a", encoding="utf-8")
        self._transcript = (self.dir / "transcript.md").open("a", encoding="utf-8")
        self._training = (self.dir / "training.jsonl").open("a", encoding="utf-8")
        self._prompts = (self.dir / "prompts.jsonl").open("a", encoding="utf-8")
        self._candidates = (self.dir / "candidates.jsonl").open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        self._log.write(message + "\n")
        self._log.flush()

    def transcript(self, markdown: str) -> None:
        self._transcript.write(markdown)
        self._transcript.flush()

    def write_config(self, payload: dict[str, Any]) -> None:
        (self.dir / "config.json").write_text(_dumps(payload), encoding="utf-8")

    def write_result(self, payload: dict[str, Any]) -> None:
        (self.dir / "result.json").write_text(_dumps(payload), encoding="utf-8")

    def write_training_epoch(self, row: dict[str, Any]) -> None:
        self._training.write(_compact(row) + "\n")
        self._training.flush()

    def write_prompt(self, row: dict[str, Any]) -> None:
        self._prompts.write(_compact(row) + "\n")
        self._prompts.flush()

    def write_candidate(self, row: dict[str, Any]) -> None:
        self._candidates.write(_compact(row) + "\n")
        self._candidates.flush()

    def close(self) -> None:
        self._log.close()
        self._transcript.close()
        self._training.close()
        self._prompts.close()
        self._candidates.close()


class NullWriter:
    """No-op writer used when no output directory is requested."""

    def log(self, message: str) -> None:
        pass

    def transcript(self, markdown: str) -> None:
        pass

    def write_config(self, payload: dict[str, Any]) -> None:
        pass

    def write_result(self, payload: dict[str, Any]) -> None:
        pass

    def write_training_epoch(self, row: dict[str, Any]) -> None:
        pass

    def write_prompt(self, row: dict[str, Any]) -> None:
        pass

    def write_candidate(self, row: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=_json_default)


def _compact(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)
