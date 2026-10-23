"""End-to-end: a REAL Strands Agent + our DeciderGate + a scripted model + a fake Decider server.

No Bedrock and no Decider weights are used. The scripted model plays the eager
agent: it "guesses" an instance id and calls terminate_instance. What this proves
is that the gate's verdicts really stop, allow or pause a tool call inside
Strands. It does not prove Decider 2B produces the right probabilities.
"""

import json
import os

import pytest

pytest.importorskip("strands")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from strands import Agent
from strands.models import Model

from decider_guard import DeciderClient, DeciderGate, GatePolicy
from opsgate import tools
from opsgate.approval import run_with_approval
from opsgate.tools import ALL_TOOLS, READ_ONLY
from tests.test_gate import FakeDecider

DEV = "i-0a1b2c3d4e5f60004"


class Scripted(Model):
    """Replays a fixed list of model turns: ("tool", name, args) or ("text", str)."""

    def __init__(self, script):
        self.script = list(script)
        self.config = {"model_id": "scripted"}

    def update_config(self, **cfg):
        self.config.update(cfg)

    def get_config(self):
        return self.config

    async def structured_output(self, *a, **k):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        kind, *rest = self.script.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        if kind == "tool":
            name, args = rest
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": rest[0]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


@pytest.fixture(autouse=True)
def fresh_fleet():
    tools._FLEET[DEV]["state"] = "running"
    yield
    tools._FLEET[DEV]["state"] = "running"


@pytest.fixture
def decider():
    servers = []

    def make(**answers):
        f = FakeDecider(answers)
        servers.append(f)
        return f

    yield make
    for s in servers:
        s.close()


def make_agent(decider_server, seen):
    gate = DeciderGate(
        client=DeciderClient(decider_server.url),
        policy=GatePolicy(skip=set(READ_ONLY)),
        on_verdict=lambda tool, args, v: seen.append(v.action),
    )
    model = Scripted([("tool", "terminate_instance", {"instance_id": DEV}), ("text", "done")])
    return Agent(model=model, tools=ALL_TOOLS, interventions=[gate], callback_handler=None)


def test_without_gate_the_eager_model_terminates_a_guessed_instance():
    model = Scripted([("tool", "terminate_instance", {"instance_id": DEV}), ("text", "done")])
    Agent(model=model, tools=ALL_TOOLS, callback_handler=None)("terminate that dev box")
    assert tools._FLEET[DEV]["state"] == "terminated"


def test_gate_stops_a_guessed_instance_id(decider):
    seen = []
    agent = make_agent(decider(grounded=0.05, premature=0.4, destructive=0.9), seen)
    agent("terminate that dev box")
    assert seen == ["guide"]
    assert tools._FLEET[DEV]["state"] == "running"


def test_gate_lets_a_safe_grounded_call_run(decider):
    seen = []
    agent = make_agent(decider(grounded=0.97, premature=0.02, destructive=0.03), seen)
    agent(f"terminate {DEV}")
    assert seen == ["proceed"]
    assert tools._FLEET[DEV]["state"] == "terminated"


def test_destructive_call_pauses_for_a_human(decider):
    seen = []
    agent = make_agent(decider(grounded=0.97, premature=0.02, destructive=0.96), seen)
    result = agent(f"terminate {DEV}")
    assert seen == ["confirm"]
    assert result.stop_reason == "interrupt"
    assert tools._FLEET[DEV]["state"] == "running"


def test_human_approval_resumes_and_runs_the_call(decider):
    seen, asked = [], []
    agent = make_agent(decider(grounded=0.97, premature=0.02, destructive=0.96), seen)
    run_with_approval(agent, f"terminate {DEV}", ask=lambda reason: asked.append(reason) or "y")
    assert len(asked) == 1 and "destructive" in asked[0]
    assert tools._FLEET[DEV]["state"] == "terminated"


def test_human_denial_resumes_without_running_the_call(decider):
    seen = []
    agent = make_agent(decider(grounded=0.97, premature=0.02, destructive=0.96), seen)
    run_with_approval(agent, f"terminate {DEV}", ask=lambda reason: "n")
    assert tools._FLEET[DEV]["state"] == "running"
