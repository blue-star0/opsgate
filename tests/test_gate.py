"""Tests run against a fake Decider HTTP server, so no GPU or model download is needed.

They verify our code (client, state rendering, policy, handler wiring). They do
NOT measure Decider 2B's accuracy: run bench/latency.py against a real server for that.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from strands.interventions import Confirm, Guide, Proceed

from decider_guard import DeciderClient, DeciderError, DeciderGate, GatePolicy, decide
from decider_guard.gate import DEFAULT_QUESTIONS, render_state


class FakeDecider:
    """Serves canned noul probabilities and records the requests it received."""

    def __init__(self, answers):
        self.answers = answers
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                out = {
                    "answers": {
                        k: {"type": "noul", "noul": outer.answers[k]} for k in body["questions"]
                    },
                    "latency_ms": 111.0,
                }
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


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


def event(tool, args, user_text):
    agent = SimpleNamespace(messages=[{"role": "user", "content": [{"text": user_text}]}])
    return SimpleNamespace(agent=agent, tool_use={"name": tool, "input": args})


def test_client_batches_all_questions_in_one_request(fake):
    f = fake(grounded=0.9, premature=0.1, destructive=0.2)
    probs = DeciderClient(f.url).ask("state", DEFAULT_QUESTIONS)
    assert probs == {"grounded": 0.9, "premature": 0.1, "destructive": 0.2}
    assert len(f.requests) == 1
    assert set(f.requests[0]["questions"]) == {"grounded", "premature", "destructive"}


def test_client_raises_when_server_down():
    with pytest.raises(DeciderError):
        DeciderClient("http://127.0.0.1:1", timeout=0.5).ask("s", DEFAULT_QUESTIONS)


def test_render_state_contains_conversation_and_call():
    s = render_state([{"role": "user", "content": [{"text": "stop the sandbox"}]}], "stop_instance", {"instance_id": "i-1"})
    assert "USER: stop the sandbox" in s
    assert 'PROPOSED TOOL CALL: stop_instance({"instance_id": "i-1"})' in s


@pytest.mark.parametrize(
    "g,p,d,expected",
    [
        (0.95, 0.05, 0.05, "proceed"),
        (0.10, 0.05, 0.05, "guide"),  # ungrounded id
        (0.95, 0.90, 0.05, "guide"),  # premature
        (0.95, 0.05, 0.95, "confirm"),  # destructive
        (0.95, 0.05, 0.50, "confirm"),  # model unsure about destructiveness -> be safe
        (0.55, 0.05, 0.05, "guide"),  # model unsure about grounding -> ask
    ],
)
def test_decide_policy(g, p, d, expected):
    v = decide({"grounded": g, "premature": p, "destructive": d}, GatePolicy(), "t", {})
    assert v.action == expected


def test_destructive_action_guide_mode():
    pol = GatePolicy(destructive_action="guide")
    v = decide({"grounded": 0.95, "premature": 0.05, "destructive": 0.95}, pol, "t", {})
    assert v.action == "guide"


def test_gate_guides_on_guessed_instance_id(fake):
    f = fake(grounded=0.05, premature=0.4, destructive=0.9)
    gate = DeciderGate(client=DeciderClient(f.url))
    r = gate.before_tool_call(event("terminate_instance", {"instance_id": "i-guess"}, "kill that server"))
    assert isinstance(r, Guide)
    assert "not stated by the user" in r.feedback
    assert "kill that server" in f.requests[0]["state"]


def test_gate_proceeds_on_safe_grounded_call(fake):
    f = fake(grounded=0.97, premature=0.02, destructive=0.03)
    gate = DeciderGate(client=DeciderClient(f.url))
    r = gate.before_tool_call(event("stop_instance", {"instance_id": "i-1"}, "stop i-1 please"))
    assert isinstance(r, Proceed)
    assert gate.history[0][1].latency_ms == 111.0


def test_gate_asks_human_to_confirm_destructive(fake):
    f = fake(grounded=0.97, premature=0.02, destructive=0.96)
    gate = DeciderGate(client=DeciderClient(f.url))
    r = gate.before_tool_call(event("terminate_instance", {"instance_id": "i-1"}, "terminate i-1"))
    assert isinstance(r, Confirm)


def test_skip_list_makes_no_decider_call(fake):
    f = fake(grounded=0, premature=1, destructive=1)
    gate = DeciderGate(client=DeciderClient(f.url), policy=GatePolicy(skip={"list_instances"}))
    assert isinstance(gate.before_tool_call(event("list_instances", {}, "list")), Proceed)
    assert f.requests == []


def test_fail_closed_when_decider_unreachable():
    gate = DeciderGate(client=DeciderClient("http://127.0.0.1:1", timeout=0.5))
    assert gate.on_error == "deny"
    with pytest.raises(DeciderError):
        gate.before_tool_call(event("stop_instance", {"instance_id": "i-1"}, "x"))


def test_fail_open_option():
    gate = DeciderGate(client=DeciderClient("http://127.0.0.1:1", timeout=0.5), policy=GatePolicy(fail_closed=False))
    assert gate.on_error == "proceed"
    assert isinstance(gate.before_tool_call(event("stop_instance", {}, "x")), Proceed)
