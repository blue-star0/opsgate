"""Build the OpsGate agent: Amazon Bedrock for language, Decider 2B for decisions.

Two Decider 2B decisions are available:
  1. Guardrails: vet every tool call (always on unless --no-gate).
  2. Model routing: pick a fast or frontier Bedrock model per turn (opt in).
"""

from __future__ import annotations

import os

from strands import Agent
from strands.models import BedrockModel

from decider_guard import DeciderClient, DeciderGate, GatePolicy, ModelRouter

from .routed import RoutedAgent
from .tools import ALL_TOOLS, READ_ONLY

# Deliberately eager. Without the gate this prompt makes the model guess
# instance ids and act, which is exactly the failure the gate exists to catch.
SYSTEM_PROMPT = (
    "You are OpsGate, an AWS operations assistant. Be fast and decisive. If the user "
    "gives an incomplete instruction, make your best guess and call the tool "
    "instead of asking questions."
)


def _assemble(model, gate: bool, on_verdict, decider_url):
    handler = None
    interventions = []
    if gate:
        handler = DeciderGate(
            client=DeciderClient(decider_url),
            policy=GatePolicy(skip=set(READ_ONLY)),
            on_verdict=on_verdict,
        )
        interventions.append(handler)

    agent = Agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        interventions=interventions,
    )
    return agent, handler


def build_agent(
    gate: bool = True,
    on_verdict=None,
    model_id: str | None = None,
    decider_url: str | None = None,
) -> tuple[Agent, DeciderGate | None]:
    chosen = model_id or os.environ.get("OPSGATE_MODEL_ID")
    # No id given: use the Strands default Bedrock model.
    model = BedrockModel(model_id=chosen) if chosen else BedrockModel()
    return _assemble(model, gate, on_verdict, decider_url)


def build_routed_agent(
    gate: bool = True,
    on_verdict=None,
    on_route=None,
    fast_model_id: str | None = None,
    frontier_model_id: str | None = None,
    decider_url: str | None = None,
) -> tuple[RoutedAgent, DeciderGate | None]:
    """Same agent, plus a Decider 2B model router in front of every turn.

    There is deliberately no default pair of model ids: which Bedrock models you
    can call depends on your account and region. Pass them or set
    OPSGATE_FAST_MODEL_ID and OPSGATE_FRONTIER_MODEL_ID.
    """
    fast = fast_model_id or os.environ.get("OPSGATE_FAST_MODEL_ID")
    frontier = frontier_model_id or os.environ.get("OPSGATE_FRONTIER_MODEL_ID")
    if not fast or not frontier:
        raise ValueError(
            "Routing needs two Bedrock model ids you have access to: set "
            "OPSGATE_FAST_MODEL_ID and OPSGATE_FRONTIER_MODEL_ID (or pass "
            "--fast-model-id and --frontier-model-id)."
        )
    model = BedrockModel(model_id=frontier)  # start on the strong model
    agent, handler = _assemble(model, gate, on_verdict, decider_url)
    router = ModelRouter(fast, frontier, client=DeciderClient(decider_url))
    return RoutedAgent(agent, model, router, on_route=on_route), handler
