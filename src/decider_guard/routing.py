"""Second reusable pattern: route each request to a fast or a frontier model.

Pattern: "Model routing". Before the LLM is called, ask Decider 2B ONE yes/no
question about the user's request ("does this need real reasoning?"), then turn
the probability into a model choice with ordinary code.

    from decider_guard import ModelRouter

    router = ModelRouter(fast_model_id="<cheap bedrock id>", frontier_model_id="<strong bedrock id>")
    decision = router.decide("what's running?")
    print(decision.tier, decision.model_id, decision.p_needs_reasoning)

The policy leans toward quality: only a confident "no" sends a request to the
fast model. Unsure answers, and a Decider outage, go to the frontier model.
(Compare the guardrail gate, which fails closed: here "closed" means "use the
strong model", because a wrong cheap answer costs more than a few extra tokens.)

This module needs no Strands imports, so it can be copied on its own together
with client.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .client import DeciderClient, DeciderError, Question


class Tier(str, Enum):
    FAST = "fast"
    FRONTIER = "frontier"


# One short question with explicit true/false boundaries. The strands-decider
# authors say questions are read less than documents, so keep it plain.
ROUTING_QUESTION = Question(
    "needs_reasoning",
    "Does answering this request need multi-step reasoning, diagnosis across several facts, or a judgment call?",
    criteria={
        "true": "the request asks to diagnose, compare, plan or decide something, or to weigh trade-offs or risk",
        "false": "the request is a lookup, a status check, or one well-defined action with every detail given",
    },
)


@dataclass
class RoutingPolicy:
    """Thresholds that turn P(needs reasoning) into a tier. Hand-picked, tune them.

    A yes/no (noul) answer has no confidence field: its probability is the
    uncertainty. So the band between `fast_max` and `frontier_min` is treated as
    "unsure" and sent to `unsure_tier`.
    """

    fast_max: float = 0.30  # P at or below this -> FAST
    frontier_min: float = 0.70  # P at or above this -> FRONTIER
    unsure_tier: Tier = Tier.FRONTIER  # P in between
    fail_tier: Tier = Tier.FRONTIER  # Decider unreachable or answered badly

    def __post_init__(self) -> None:
        if not (0.0 <= self.fast_max < self.frontier_min <= 1.0):
            raise ValueError("need 0 <= fast_max < frontier_min <= 1")


@dataclass
class RouteDecision:
    tier: Tier
    model_id: str
    p_needs_reasoning: float | None  # None when Decider failed
    reason: str
    latency_ms: float | None = None


def route(p_needs_reasoning: float | None, policy: RoutingPolicy) -> Tier:
    """Pure function: probability -> tier. No model, no network. Easy to test."""
    if p_needs_reasoning is None:
        return policy.fail_tier
    if p_needs_reasoning >= policy.frontier_min:
        return Tier.FRONTIER
    if p_needs_reasoning <= policy.fast_max:
        return Tier.FAST
    return policy.unsure_tier


def _text_of(message: dict) -> str:
    parts = []
    for block in message.get("content", []) or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return " ".join(parts)


def render_route_state(user_message: str, messages: list[dict] | None = None, max_turns: int = 4) -> str:
    """Plain-text state for Decider: a little recent context plus the new request."""
    lines = []
    for m in (messages or [])[-max_turns:]:
        text = _text_of(m).strip()
        if text:
            lines.append(f"{m.get('role', 'unknown').upper()}: {text}")
    lines.append(f"NEW USER REQUEST: {user_message}")
    return "\n".join(lines)


@dataclass
class ModelRouter:
    """Chooses a Bedrock model id for each user turn using one Decider 2B question."""

    fast_model_id: str
    frontier_model_id: str
    client: DeciderClient = field(default_factory=DeciderClient)
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    question: Question = ROUTING_QUESTION
    history: list[RouteDecision] = field(default_factory=list)

    def model_for(self, tier: Tier) -> str:
        return self.fast_model_id if tier is Tier.FAST else self.frontier_model_id

    def decide(self, user_message: str, messages: list[dict] | None = None) -> RouteDecision:
        state = render_route_state(user_message, messages)
        try:
            p = self.client.ask(state, [self.question])[self.question.name]
        except DeciderError as exc:
            tier = self.policy.fail_tier
            decision = RouteDecision(
                tier, self.model_for(tier), None, f"Decider unavailable ({exc}); using the {tier.value} model"
            )
        else:
            tier = route(p, self.policy)
            decision = RouteDecision(
                tier,
                self.model_for(tier),
                p,
                f"needs_reasoning={p:.2f} -> {tier.value}",
                self.client.last_latency_ms,
            )
        self.history.append(decision)
        return decision

    def fast_share(self) -> float | None:
        """Fraction of decisions sent to the fast model. Measure it before claiming savings."""
        if not self.history:
            return None
        return sum(d.tier is Tier.FAST for d in self.history) / len(self.history)
