"""Measure how Decider 2B routes a small labelled set of requests, and how fast.

Usage (Decider server must be running; no AWS calls are made):
    python bench/routing.py

The labels below are the author's own judgement of "easy" vs "hard", and the set
is tiny. Treat the output as a smoke test of the routing question, not as an
accuracy claim. Replace PROMPTS with real traffic before quoting any number, and
measure the fast-model share on that traffic before claiming a cost saving.
"""

import argparse
import json
import statistics

from decider_guard import DeciderClient, ModelRouter, Tier

# (request, expected tier)
PROMPTS = [
    ("what's running?", Tier.FAST),
    ("list my instances", Tier.FAST),
    ("is dev-sandbox running?", Tier.FAST),
    ("stop i-0a1b2c3d4e5f60004", Tier.FAST),
    ("show the state of i-0a1b2c3d4e5f60005", Tier.FAST),
    ("why is db-prod slow, and should I resize it or add a read replica?", Tier.FRONTIER),
    ("compare keeping web-prod-1 and web-prod-2 on t3.medium against one larger instance", Tier.FRONTIER),
    ("which instances look unused, and what is the risk of stopping each one?", Tier.FRONTIER),
    ("plan a safe order to shut down the dev stack this weekend", Tier.FRONTIER),
    ("should we terminate old-batch-runner or keep it for the quarterly job?", Tier.FRONTIER),
]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="repeats of the whole set (after one warmup)")
    a = ap.parse_args()

    router = ModelRouter("fast", "frontier", client=DeciderClient())
    router.client.ask("warmup", [router.question])  # first call loads kernels

    wall, rows = [], []
    for _ in range(a.n):
        for text, expected in PROMPTS:
            d = router.decide(text)
            rows.append((text, expected, d))
            if d.latency_ms is not None:
                wall.append(d.latency_ms)

    last = rows[-len(PROMPTS):]
    for text, expected, d in last:
        mark = "ok " if d.tier is expected else "MISS"
        p = "n/a" if d.p_needs_reasoning is None else f"{d.p_needs_reasoning:.2f}"
        print(f"{mark} p={p} -> {d.tier.value:8s} (expected {expected.value:8s}) {text}")

    agree = sum(d.tier is e for _, e, d in rows) / len(rows)
    out = {
        "requests": len(rows),
        "agreement_with_author_labels": round(agree, 2),
        "fast_share": round(router.fast_share(), 2),
        "decider_latency_ms": {"median": round(statistics.median(wall)), "p95": round(pct(wall, 0.95))} if wall else None,
    }
    print(json.dumps(out, indent=2))
