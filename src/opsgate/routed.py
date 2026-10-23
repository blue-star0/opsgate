"""A thin wrapper that picks the Bedrock model for each turn, then runs the agent."""

from __future__ import annotations

from typing import Any, Callable

from decider_guard import ModelRouter, RouteDecision


class RoutedAgent:
    """Call it like an Agent. Before each new user turn, Decider 2B chooses the model id.

    The agent, its tools and its Decider gate are unchanged. Only the model id of
    the single BedrockModel is switched between turns, so the conversation
    history carries over.
    """

    def __init__(
        self,
        agent: Any,
        model: Any,
        router: ModelRouter,
        on_route: Callable[[RouteDecision], None] | None = None,
    ):
        self.agent = agent
        self.model = model
        self.router = router
        self.on_route = on_route

    def _use_model(self, model_id: str) -> None:
        update = getattr(self.model, "update_config", None)
        if not callable(update):
            raise RuntimeError("This Strands model has no update_config(); cannot switch models per turn.")
        update(model_id=model_id)

    def __call__(self, prompt: Any, **kwargs: Any) -> Any:
        # A non-string prompt is a resume after a human approval, not a new request:
        # keep the model that is already working on it.
        if isinstance(prompt, str):
            decision = self.router.decide(prompt, getattr(self.agent, "messages", None))
            self._use_model(decision.model_id)
            if self.on_route:
                self.on_route(decision)
        return self.agent(prompt, **kwargs)
