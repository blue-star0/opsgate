"""decider_guard: reusable Strands Decider 2B decision patterns for Strands agents.

Pattern 1, "Guardrails" (gate.py): before a tool call runs, ask Decider 2B a few
narrow yes/no questions about the proposed call, then turn the probabilities into
a typed Strands intervention action (Proceed / Guide / Confirm).

Pattern 2, "Model routing" (routing.py): before the LLM runs, ask Decider 2B one
yes/no question about the request, then pick a fast or a frontier model.
"""

from .client import DeciderClient, DeciderError, Question
from .gate import DeciderGate, GatePolicy, Verdict, decide
from .routing import ModelRouter, RouteDecision, RoutingPolicy, Tier, route

__all__ = [
    "DeciderClient",
    "DeciderError",
    "Question",
    "DeciderGate",
    "GatePolicy",
    "Verdict",
    "decide",
    "ModelRouter",
    "RouteDecision",
    "RoutingPolicy",
    "Tier",
    "route",
]
