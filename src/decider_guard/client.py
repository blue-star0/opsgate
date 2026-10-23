"""Tiny HTTP client for a locally served Strands Decider.

Start the server with:
    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

Request shape (from the strands-decider README):
    POST /v1/systemone
    {"state": "...", "questions": {"name": {"type": "noul", "instructions": "..."}}}

Response shape for a noul question:
    {"answers": {"name": {"type": "noul", "noul": 0.858}}, "latency_ms": 489.0, ...}

Closer to 1.0 leans Yes. Only `noul` (yes/no) questions are used here because
that is the response shape the README documents for the HTTP endpoint.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_URL = "http://127.0.0.1:8099"


class DeciderError(RuntimeError):
    """Raised when the Decider server cannot be reached or answers badly."""


@dataclass(frozen=True)
class Question:
    """A yes/no question.

    `instructions` is what the model is asked. `criteria` is optional and maps
    {"true": "...", "false": "..."} to short descriptions that sharpen the boundary
    (supported by the strands-decider server's NoulQuestion schema).
    """

    name: str
    instructions: str
    criteria: dict[str, str] | None = None


def _noul_body(q: Question) -> dict:
    body: dict = {"type": "noul", "instructions": q.instructions}
    if q.criteria:
        body["criteria"] = q.criteria
    return body


class DeciderClient:
    def __init__(self, url: str | None = None, timeout: float = 5.0):
        self.url = (url or os.environ.get("STRANDS_DECIDER_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.last_latency_ms: float | None = None

    def ask(self, state: str, questions: list[Question]) -> dict[str, float]:
        """Ask several yes/no questions about one state in a single request.

        Returns {question_name: probability_of_yes}. The state is read once by
        the model, so extra questions are cheap.
        """
        body = {
            "state": state,
            "questions": {q.name: _noul_body(q) for q in questions},
        }
        req = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DeciderError(f"Decider server at {self.url} failed: {exc}") from exc

        self.last_latency_ms = payload.get("latency_ms")
        out: dict[str, float] = {}
        for q in questions:
            ans = payload.get("answers", {}).get(q.name)
            if not ans or "noul" not in ans:
                raise DeciderError(f"No noul answer for question '{q.name}': {payload}")
            out[q.name] = float(ans["noul"])
        return out
