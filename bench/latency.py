"""Measure Decider 2B latency, and compare the same decision made by an LLM on Bedrock.

Usage (Decider server must be running):
    python bench/latency.py                 # Decider only
    python bench/latency.py --llm           # also time the same 3 questions via Bedrock

Numbers depend on your hardware and region. Report what you measure, not what the blog says.
"""

import argparse
import json
import statistics
import time

from decider_guard import DeciderClient
from decider_guard.gate import DEFAULT_QUESTIONS, render_state

CASES = [
    ([{"role": "user", "content": [{"text": "Terminate i-0a1b2c3d4e5f60004"}]}], "terminate_instance", {"instance_id": "i-0a1b2c3d4e5f60004"}),
    ([{"role": "user", "content": [{"text": "kill that server"}]}], "terminate_instance", {"instance_id": "i-0a1b2c3d4e5f60003"}),
    ([{"role": "user", "content": [{"text": "stop the sandbox box"}]}], "stop_instance", {"instance_id": "i-0a1b2c3d4e5f60004"}),
    ([{"role": "user", "content": [{"text": "what's running?"}]}], "list_instances", {}),
]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def bench_decider(n):
    c = DeciderClient()
    c.ask("warmup", DEFAULT_QUESTIONS)  # first call loads kernels
    wall = []
    for _ in range(n):
        for msgs, tool, args in CASES:
            t = time.perf_counter()
            c.ask(render_state(msgs, tool, args), DEFAULT_QUESTIONS)
            wall.append((time.perf_counter() - t) * 1000)
    return wall


def bench_llm(n):
    import boto3

    rt = boto3.client("bedrock-runtime")
    model = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # change to a model you have access to
    prompt = (
        "Answer with JSON {grounded: bool, premature: bool, destructive: bool} only.\n"
        + "\n".join(q.instructions for q in DEFAULT_QUESTIONS)
        + "\n\n"
    )
    wall = []
    for _ in range(n):
        for msgs, tool, args in CASES:
            t = time.perf_counter()
            rt.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt + render_state(msgs, tool, args)}]}],
                inferenceConfig={"maxTokens": 60},
            )
            wall.append((time.perf_counter() - t) * 1000)
    return wall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    out = {"decider_ms": None}
    d = bench_decider(a.n)
    out["decider_ms"] = {"median": round(statistics.median(d)), "p95": round(pct(d, 0.95)), "n": len(d)}
    if a.llm:
        l = bench_llm(max(1, a.n // 3))
        out["bedrock_llm_ms"] = {"median": round(statistics.median(l)), "p95": round(pct(l, 0.95)), "n": len(l)}
    print(json.dumps(out, indent=2))
