"""The reusable decision pattern: gate a tool call on Decider 2B yes/no verdicts.

How to reuse it in your own agent:

    from decider_guard import DeciderGate
    agent = Agent(tools=[...], interventions=[DeciderGate()])

Or keep your own questions/policy by calling `decide()` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from strands.interventions import Confirm, Guide, InterventionHandler, Proceed

from .client import DeciderClient, DeciderError, Question

# The three questions asked about every proposed tool call. They are asked in
# ONE request, so the conversation is read by the model once.
#
# The strands-decider authors say questions are "read less than documents" and
# that the obvious reading should be the intended one, so each question is short
# and has explicit true/false boundary criteria.
DEFAULT_QUESTIONS = [
    Question(
        "grounded",
        "Does the user's message contain every argument value of the proposed tool call?",
        criteria={
            "true": "every argument value (ids, names, numbers) appears in the user's messages",
            "false": "at least one argument value was guessed or invented by the assistant",
        },
    ),
    Question(
        "premature",
        "Does the assistant need to ask the user one clarifying question before calling this tool?",
        criteria={
            "true": "the request is ambiguous or incomplete",
            "false": "the request is specific enough to act on now",
        },
    ),
    Question(
        "destructive",
        "Does this tool call permanently stop, delete or terminate a cloud resource?",
        criteria={
            "true": "the call terminates or deletes something, and cannot be undone",
            "false": "the call only reads or reversibly changes state",
        },
    ),
]


@dataclass
class GatePolicy:
    """Thresholds that turn probabilities into actions. Hand-picked, tune them.

    Note: the strands-decider docs' "0.9 confidence = ~95% right" figure is for
    choice/score answers. A yes/no (noul) answer has no confidence field: its
    probability is the uncertainty, so these thresholds are not calibrated claims.
    """

    grounded_min: float = 0.7  # P(grounded) below this -> Guide (ask the user)
    premature_max: float = 0.5  # P(premature) above this -> Guide (ask the user)
    destructive_min: float = 0.5  # P(destructive) at/above this -> destructive path
    unsure_band: float = 0.15  # |p - 0.5| below this means the model is unsure
    destructive_action: str = "confirm"  # "confirm" (human approves) or "guide"
    fail_closed: bool = True  # if Decider is down, do not let the call run
    always_check: set[str] = field(default_factory=set)  # empty = check every tool
    skip: set[str] = field(default_factory=set)  # tools never gated


@dataclass
class Verdict:
    action: str  # proceed | guide | confirm
    reason: str
    probs: dict[str, float]
    latency_ms: float | None = None


def _unsure(p: float, band: float) -> bool:
    return abs(p - 0.5) < band


def decide(probs: dict[str, float], policy: GatePolicy, tool: str, args: dict) -> Verdict:
    """Pure function: probabilities -> verdict. No model, no network. Easy to test."""
    g, p, d = probs["grounded"], probs["premature"], probs["destructive"]

    grounding_doubtful = g < policy.grounded_min or (_unsure(g, policy.unsure_band) and g < 0.85)
    if grounding_doubtful:
        return Verdict(
            "guide",
            f"Some arguments for '{tool}' were not stated by the user "
            f"(grounded={g:.2f}). Do not guess values. Ask the user for the exact "
            f"missing details, then call the tool again.",
            probs,
        )
    if p > policy.premature_max:
        return Verdict(
            "guide",
            f"It is too early to call '{tool}' (premature={p:.2f}). "
            f"Ask the user a short clarifying question first.",
            probs,
        )
    if d >= policy.destructive_min or _unsure(d, policy.unsure_band):
        if policy.destructive_action == "confirm":
            return Verdict(
                "confirm",
                f"'{tool}' looks destructive (destructive={d:.2f}) with arguments "
                f"{json.dumps(args, sort_keys=True)}. Approve?",
                probs,
            )
        return Verdict(
            "guide",
            f"'{tool}' looks destructive (destructive={d:.2f}). Repeat the exact "
            f"resource and action back to the user and ask them to confirm "
            f"before calling the tool.",
            probs,
        )
    return Verdict("proceed", "grounded, not premature, not destructive", probs)


def _text_of(message: dict) -> str:
    parts = []
    for block in message.get("content", []) or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return " ".join(parts)


def render_state(messages: list[dict], tool: str, args: dict, max_turns: int = 8) -> str:
    """Render recent conversation + the proposed call as plain text for Decider."""
    lines = []
    for m in messages[-max_turns:]:
        text = _text_of(m).strip()
        if text:
            lines.append(f"{m.get('role', 'unknown').upper()}: {text}")
    lines.append(f"PROPOSED TOOL CALL: {tool}({json.dumps(args, sort_keys=True)})")
    return "\n".join(lines)


class DeciderGate(InterventionHandler):
    """Strands intervention that runs Decider 2B before every tool call."""

    name = "decider-gate"

    def __init__(
        self,
        client: DeciderClient | None = None,
        policy: GatePolicy | None = None,
        questions: list[Question] | None = None,
        on_verdict: Any = None,
    ):
        self.client = client or DeciderClient()
        self.policy = policy or GatePolicy()
        self.questions = questions or DEFAULT_QUESTIONS
        self.on_verdict = on_verdict  # optional callback(tool, args, Verdict) for logging
        self.history: list[tuple[str, Verdict]] = []

    @property
    def on_error(self) -> str:
        return "deny" if self.policy.fail_closed else "proceed"

    def before_tool_call(self, event):
        tool = event.tool_use["name"]
        args = event.tool_use.get("input", {}) or {}
        if tool in self.policy.skip:
            return Proceed()

        messages = getattr(event.agent, "messages", []) or []
        state = render_state(messages, tool, args)
        try:
            probs = self.client.ask(state, self.questions)
        except DeciderError:
            if self.policy.fail_closed:
                raise  # on_error == "deny" turns this into a Deny
            return Proceed()

        verdict = decide(probs, self.policy, tool, args)
        verdict.latency_ms = self.client.last_latency_ms
        self.history.append((tool, verdict))
        if self.on_verdict:
            self.on_verdict(tool, args, verdict)

        if verdict.action == "proceed":
            return Proceed()
        if verdict.action == "confirm":
            return Confirm(prompt=verdict.reason)
        return Guide(feedback=verdict.reason)
