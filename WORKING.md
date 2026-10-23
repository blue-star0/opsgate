# How OpsGate works

## One turn, step by step

1. **You type a request**, for example `terminate that dev box`.
2. **Route (only with `--route`).** `ModelRouter` sends Decider 2B the recent conversation plus the request and asks one yes/no question: *does this need multi-step reasoning, diagnosis across several facts, or a judgment call?* Decider returns a probability. `route()` maps it to a model: at or below 0.30 the fast model, otherwise the frontier model. `RoutedAgent` then switches the single `BedrockModel` to that id.
3. **The Bedrock LLM runs.** The system prompt is deliberately eager ("make your best guess and call the tool"), so without the gate the model invents an instance id and calls `terminate_instance`.
4. **The gate runs before the tool.** Strands calls `DeciderGate.before_tool_call`. The gate renders the recent conversation plus the proposed call as plain text and sends Decider three yes/no questions in one request:
   - `grounded`: does the user's message contain every argument value?
   - `premature`: does the assistant need to ask a clarifying question first?
   - `destructive`: does the call permanently stop, delete or terminate something?
5. **`decide()` turns the three probabilities into an action**, in this order:

   | Check | Action | What happens |
   |---|---|---|
   | P(grounded) below 0.70 | `Guide` | the call is cancelled and the model is told to ask the user for the missing detail |
   | P(premature) above 0.50 | `Guide` | the call is cancelled and the model asks a clarifying question |
   | P(destructive) at or above 0.50, or within 0.15 of 0.50 (in effect, above 0.35) | `Confirm` | the turn pauses; a human types y or n |
   | otherwise | `Proceed` | the tool runs |

6. **Approval.** A `Confirm` ends the turn with `stop_reason == "interrupt"`. `run_with_approval` shows the reason, reads y or n, and resumes the turn. `y` runs the tool; anything else cancels it and the model is told the call was not confirmed.
7. **Decider down?** The gate fails closed: the call is denied. The router fails toward quality: it uses the frontier model.

## Worked example (the demo)

| You say | Model proposes | Decider says (illustrative) | Result |
|---|---|---|---|
| `terminate that dev box` | `terminate_instance(i-...0004)` | grounded low | **Guide**: "ask the user which instance". The fleet is unchanged |
| `terminate i-0a1b2c3d4e5f60004` | `terminate_instance(i-...0004)` | grounded high, destructive high | **Confirm**: you type `y`, then it runs |
| `what's running?` | `list_instances()` | not checked (read-only tools are skipped) | runs |

The probabilities in that table are illustrative. They match what the offline tests feed the gate, not measured Decider 2B output.

## File map

| File | Role |
|---|---|
| `src/decider_guard/client.py` | HTTP client for `strands-decider serve`: one request, many yes/no questions |
| `src/decider_guard/gate.py` | Pattern 1: the three questions, `GatePolicy`, `decide()`, `DeciderGate` |
| `src/decider_guard/routing.py` | Pattern 2: the routing question, `RoutingPolicy`, `route()`, `ModelRouter` |
| `src/opsgate/tools.py` | `list_instances`, `stop_instance`, `terminate_instance` over a fake fleet (or real EC2 with `OPSGATE_LIVE=1`) |
| `src/opsgate/agent.py` | Builds the Strands agent: `build_agent` and `build_routed_agent` |
| `src/opsgate/routed.py` | `RoutedAgent`: picks the model id before each new request |
| `src/opsgate/approval.py` | `run_with_approval`: asks the human when the gate returns `Confirm` |
| `src/opsgate/cli.py` | The `opsgate` command; prints every `[decider ...]` and `[route ...]` line |
| `tests/` | 44 offline tests plus 5 live tests that need a Decider server |
| `bench/latency.py` | Gate latency, optionally against a Bedrock LLM doing the same job |
| `bench/routing.py` | Routing smoke test on ten labelled requests, with latency and fast-model share |
| `deploy/` | EC2 user-data script and a Bedrock-only IAM policy (not yet run) |

## Why a separate model makes these decisions

The LLM that proposed the call would be grading itself. Decider is a separate 2B model that runs locally and never generates text. It returns a number, so the policy is ordinary code you can unit test, and the agent cannot talk its way past it. All questions in one request share one read of the conversation, so adding a question costs little.

## How to adapt it

- **New domain:** replace `DEFAULT_QUESTIONS` in `gate.py` and the criteria text. Keep each question short and give it true/false criteria.
- **New thresholds:** edit `GatePolicy` or `RoutingPolicy`. Replay your own conversations through `bench/` first.
- **Your own approval UI:** skip `run_with_approval` and resume the turn yourself with `[{"interruptResponse": {"interruptId": ..., "response": "y"}}]`.
- **Another routing pair:** pass any two Bedrock model ids you can call to `ModelRouter`.
