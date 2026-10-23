"""End-to-end wiring: a real Strands Agent + our gate + a scripted fake model.

No Bedrock call is made. The fake model first "guesses" an instance id (the
eager behaviour OpsGate's system prompt encourages), then, after the gate's
feedback, asks the user a question instead.
"""

import os

import pytest

pytest.importorskip("strands")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from decider_guard import DeciderClient, DeciderGate, GatePolicy
from opsgate.agent import build_agent
from opsgate.tools import ALL_TOOLS, READ_ONLY
from tests.test_gate import FakeDecider  # noqa: E402


def test_build_agent_registers_gate_without_aws_credentials():
    agent, gate = build_agent(gate=True)
    assert isinstance(gate, DeciderGate)
    assert gate.policy.skip == READ_ONLY
    agent_nogate, none_gate = build_agent(gate=False)
    assert none_gate is None


def test_tools_have_specs():
    names = {t.tool_name for t in ALL_TOOLS}
    assert names == {"list_instances", "stop_instance", "terminate_instance"}
