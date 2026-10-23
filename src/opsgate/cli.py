"""Interactive CLI. Shows every Decider verdict and routing choice next to the agent's answer."""

from __future__ import annotations

import argparse

from .agent import build_agent, build_routed_agent
from .approval import run_with_approval


def _print_verdict(tool: str, args: dict, v) -> None:
    p = v.probs
    lat = f"{v.latency_ms:.0f}ms" if v.latency_ms else "n/a"
    print(
        f"  [decider {lat}] {tool}({args}) -> {v.action.upper()}  "
        f"grounded={p['grounded']:.2f} premature={p['premature']:.2f} destructive={p['destructive']:.2f}"
    )


def _print_route(d) -> None:
    lat = f"{d.latency_ms:.0f}ms" if d.latency_ms else "n/a"
    print(f"  [route {lat}] {d.reason} -> {d.model_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OpsGate: AWS ops agent gated and routed by Strands Decider 2B")
    ap.add_argument("--no-gate", action="store_true", help="run WITHOUT the Decider guardrail (for the before/after demo)")
    ap.add_argument("--route", action="store_true", help="let Decider 2B pick a fast or frontier Bedrock model per turn")
    ap.add_argument("--fast-model-id", default=None, help="with --route; default: $OPSGATE_FAST_MODEL_ID")
    ap.add_argument("--frontier-model-id", default=None, help="with --route; default: $OPSGATE_FRONTIER_MODEL_ID")
    ap.add_argument("--decider-url", default=None, help="default: $STRANDS_DECIDER_URL or http://127.0.0.1:8099")
    ap.add_argument("--model-id", default=None, help="Bedrock model id when not routing (default: Strands default)")
    ap.add_argument("prompt", nargs="*", help="one-shot prompt; omit for interactive mode")
    a = ap.parse_args()

    if a.route:
        try:
            agent, _ = build_routed_agent(
                gate=not a.no_gate,
                on_verdict=_print_verdict,
                on_route=_print_route,
                fast_model_id=a.fast_model_id,
                frontier_model_id=a.frontier_model_id,
                decider_url=a.decider_url,
            )
        except ValueError as exc:
            ap.error(str(exc))
    else:
        agent, _ = build_agent(
            gate=not a.no_gate,
            on_verdict=_print_verdict,
            model_id=a.model_id,
            decider_url=a.decider_url,
        )
    mode = "(UNGATED)" if a.no_gate else "(gated by Strands Decider 2B)"
    print("OpsGate", mode, "+ model routing" if a.route else "", "- Ctrl-C to quit")

    if a.prompt:
        run_with_approval(agent, " ".join(a.prompt))
        return
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if msg:
            run_with_approval(agent, msg)


if __name__ == "__main__":
    main()
