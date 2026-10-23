"""Human approval for `Confirm` verdicts.

When the Decider gate returns Confirm, Strands pauses the turn and returns a result
with stop_reason == "interrupt". Something has to ask the human and resume the
turn, or the destructive call silently never happens. This is that something.
"""

from __future__ import annotations

from typing import Any, Callable


def ask_on_terminal(reason: str) -> str:
    print(f"\n  [approval needed] {reason}")
    try:
        return input("  approve? [y/N] ").strip() or "n"
    except (EOFError, KeyboardInterrupt):
        return "n"  # no human at the keyboard means no approval


def run_with_approval(agent: Any, prompt: str, ask: Callable[[str], str] = ask_on_terminal) -> Any:
    """Run one turn, asking `ask(reason)` whenever the gate pauses for approval.

    Strands treats "y" / "yes" / True as approval; anything else cancels the tool
    call and tells the model it was not confirmed.
    """
    result = agent(prompt)
    while getattr(result, "stop_reason", None) == "interrupt":
        responses = [
            {"interruptResponse": {"interruptId": it.id, "response": ask(it.reason)}}
            for it in result.interrupts
        ]
        result = agent(responses)
    return result
