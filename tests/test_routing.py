"""Model-routing tests. Pure policy tests plus a fake Decider HTTP server, so no GPU,
model download or AWS credentials are needed.

They verify our code (policy, router, model switching). They do NOT measure
Decider 2B's accuracy at telling easy requests from hard ones: run
bench/routing.py against a real server for that.
"""

import pytest

from decider_guard import DeciderClient, ModelRouter, RoutingPolicy, Tier, route
from decider_guard.routing import ROUTING_QUESTION, render_route_state
from tests.test_gate import FakeDecider


@pytest.fixture
def fake():
    servers = []

    def make(**answers):
        f = FakeDecider(answers)
        servers.append(f)
        return f

    yield make
    for s in servers:
        s.close()


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.02, Tier.FAST),
        (0.30, Tier.FAST),  # boundary is inclusive
        (0.31, Tier.FRONTIER),  # unsure band -> quality
        (0.50, Tier.FRONTIER),
        (0.69, Tier.FRONTIER),
        (0.70, Tier.FRONTIER),
        (0.98, Tier.FRONTIER),
        (None, Tier.FRONTIER),  # Decider failed -> fail tier
    ],
)
def test_route_policy(p, expected):
    assert route(p, RoutingPolicy()) is expected


def test_unsure_band_and_fail_tier_are_configurable():
    pol = RoutingPolicy(unsure_tier=Tier.FAST, fail_tier=Tier.FAST)
    assert route(0.5, pol) is Tier.FAST
    assert route(None, pol) is Tier.FAST
    assert route(0.9, pol) is Tier.FRONTIER


def test_policy_rejects_crossed_thresholds():
    with pytest.raises(ValueError):
        RoutingPolicy(fast_max=0.8, frontier_min=0.2)


def test_render_route_state_has_context_and_request():
    s = render_route_state("what's running?", [{"role": "user", "content": [{"text": "hi"}]}])
    assert "USER: hi" in s
    assert "NEW USER REQUEST: what's running?" in s


def test_router_sends_easy_request_to_fast_model(fake):
    f = fake(needs_reasoning=0.05)
    r = ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url))
    d = r.decide("what's running?")
    assert d.tier is Tier.FAST and d.model_id == "fast-id"
    assert d.p_needs_reasoning == 0.05
    assert d.latency_ms == 111.0
    assert len(f.requests) == 1
    assert set(f.requests[0]["questions"]) == {ROUTING_QUESTION.name}
    assert "what's running?" in f.requests[0]["state"]


def test_router_sends_hard_request_to_frontier_model(fake):
    f = fake(needs_reasoning=0.95)
    r = ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url))
    d = r.decide("why is db-prod slow and should I resize it?")
    assert d.tier is Tier.FRONTIER and d.model_id == "strong-id"


def test_router_prefers_quality_when_unsure(fake):
    f = fake(needs_reasoning=0.5)
    r = ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url))
    assert r.decide("stop the sandbox").model_id == "strong-id"


def test_router_fails_toward_the_strong_model():
    r = ModelRouter("fast-id", "strong-id", client=DeciderClient("http://127.0.0.1:1", timeout=0.5))
    d = r.decide("what's running?")
    assert d.tier is Tier.FRONTIER and d.model_id == "strong-id"
    assert d.p_needs_reasoning is None
    assert "unavailable" in d.reason


def test_fast_share_is_measured_not_assumed(fake):
    f = fake(needs_reasoning=0.05)
    r = ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url))
    assert r.fast_share() is None
    r.decide("a")
    r.decide("b")
    assert r.fast_share() == 1.0
    f.answers["needs_reasoning"] = 0.9
    r.decide("c")
    r.decide("d")
    assert r.fast_share() == 0.5


# --- RoutedAgent: switches the model id, then runs the agent (no Strands, no Bedrock) ---


class _Model:
    def __init__(self):
        self.model_ids = []

    def update_config(self, **cfg):
        self.model_ids.append(cfg["model_id"])


class _Agent:
    def __init__(self, model):
        self.messages = []
        self.model = model
        self.seen = []

    def __call__(self, prompt, **kwargs):
        self.seen.append((prompt, self.model.model_ids[-1]))
        return "ok"


def test_routed_agent_switches_model_before_each_turn(fake):
    from opsgate.routed import RoutedAgent

    f = fake(needs_reasoning=0.05)
    model = _Model()
    agent = _Agent(model)
    routes = []
    ra = RoutedAgent(agent, model, ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url)), on_route=routes.append)

    assert ra("what's running?") == "ok"
    f.answers["needs_reasoning"] = 0.95
    ra("why is db-prod slow?")

    assert agent.seen == [("what's running?", "fast-id"), ("why is db-prod slow?", "strong-id")]
    assert [d.tier for d in routes] == [Tier.FAST, Tier.FRONTIER]


def test_routed_agent_errors_clearly_if_model_cannot_switch():
    from opsgate.routed import RoutedAgent

    ra = RoutedAgent(_Agent(object()), object(), ModelRouter("a", "b", client=DeciderClient("http://127.0.0.1:1", timeout=0.5)))
    with pytest.raises(RuntimeError, match="update_config"):
        ra("hi")


def test_build_routed_agent_needs_two_model_ids(monkeypatch):
    pytest.importorskip("strands")
    from opsgate.agent import build_routed_agent

    monkeypatch.delenv("OPSGATE_FAST_MODEL_ID", raising=False)
    monkeypatch.delenv("OPSGATE_FRONTIER_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="OPSGATE_FAST_MODEL_ID"):
        build_routed_agent()


def test_routed_agent_does_not_reroute_a_resume_after_approval(fake):
    from opsgate.routed import RoutedAgent

    f = fake(needs_reasoning=0.05)
    model = _Model()
    model.update_config(model_id="strong-id")
    agent = _Agent(model)
    ra = RoutedAgent(agent, model, ModelRouter("fast-id", "strong-id", client=DeciderClient(f.url)))

    ra([{"interruptResponse": {"interruptId": "x", "response": "y"}}])  # resume, not a new request

    assert f.requests == []  # Decider was not asked
    assert model.model_ids == ["strong-id"]  # model unchanged
