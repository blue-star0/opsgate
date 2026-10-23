# OpsGate: an AWS ops agent whose decisions are made by Strands Decider 2B

**Declared decision pattern: Guardrails.** Before each tool call runs, Strands Decider 2B answers three yes/no questions about the proposed call: are the arguments grounded in what the user said, is a clarifying question needed first, and would the call irreversibly delete or terminate something?

**Second reusable pattern: Model routing.** Before each turn, Decider 2B answers one yes/no question about the request ("does this need real reasoning?") and the agent picks a fast or a frontier Amazon Bedrock model.

Amazon Bedrock writes the language and proposes the tool call. Decider 2B makes the yes/no decisions locally, in front of the LLM and in front of every tool.

```
                  ┌────────────────────────────────────────────────────────────┐
user request ──►  │ [ Decider 2B router ]  needs_reasoning?   (opt in: --route) │
                  └───────────────┬────────────────────────────────────────────┘
                     low ─► fast model        high / unsure / Decider down ─► frontier model
                                  │
                         Bedrock LLM proposes a tool call
                                  │
                  ┌───────────────▼────────────────────────────────────────────┐
                  │ [ Decider 2B gate ]  grounded?  premature?  destructive?   │
                  └───┬──────────────────┬──────────────────┬──────────────────┘
                  proceed             guide              confirm
                  run the tool   model asks the user   a human approves (y/N)
                                 instead of guessing   before it runs
```

How a turn works in detail, with a file map: [WORKING.md](WORKING.md).

## Status (read this first)

What has and has not been checked. Nothing below is rounded up.

| Piece | State | Evidence |
|---|---|---|
| Decider client matches the server API | Checked against the server source | strands-decider `e69edb2`: `POST /v1/systemone`, a yes/no answer is `{"type":"noul","noul": P(true)}` |
| Gate policy, routing policy, state rendering | Tested | Unit tests, no model needed |
| Gate inside a real Strands agent | Tested | `tests/test_e2e_strands.py`: a real `strands.Agent` with a scripted model and a fake Decider server. A guessed id is stopped, a safe call runs, a destructive call pauses, and human approval or denial resumes correctly (strands-agents 1.62.0) |
| Per-turn model switching | Tested | `BedrockModel.update_config(model_id=...)` changes the model the agent holds |
| Test suite | 44 pass, 5 skipped | The 5 skipped are the live tests that need a real Decider server |
| **Real Decider 2B answers** | **Not run** | The weights are on Hugging Face and the build sandbox could not reach them, so no real model output exists yet. Whether Decider scores guessed ids low, or hard requests high, is unmeasured |
| **Bedrock end to end** | **Not run** | No AWS credentials in the build sandbox |
| **Latency numbers** | **Not measured** | `bench/latency.py` and `bench/routing.py` are ready |
| **Cost saving from routing** | **Not measured** | It depends on how much of your traffic Decider sends to the fast model. `bench/routing.py` reports that share |
| EC2 deploy script, IAM policy | Not run | |

To close the gaps, on a machine with Hugging Face access and AWS credentials:

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
export STRANDS_DECIDER_URL=http://127.0.0.1:8099
pytest tests/test_live_decider.py -v     # real Decider answers
python bench/latency.py                  # gate latency (add --llm to compare a Bedrock LLM)
python bench/routing.py                  # routing smoke test and latency
```

## Quick start

```bash
# 1. Decider 2B, served locally (GPU, Apple silicon, or CPU)
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099

# 2. OpsGate (needs AWS credentials with Bedrock model access)
pip install -e .
export OPSGATE_MODEL_ID=<a Bedrock model id you can call>   # optional; Strands default otherwise
opsgate --no-gate "terminate that dev box"     # BEFORE: the eager model guesses an id and acts
opsgate "terminate that dev box"               # AFTER: the gate stops the guess

# 3. Optional: model routing (you choose the two models)
export OPSGATE_FAST_MODEL_ID=<cheap Bedrock model id>
export OPSGATE_FRONTIER_MODEL_ID=<strong Bedrock model id>
opsgate --route "what's running?"
opsgate --route "why is db-prod slow, and should I resize it?"
```

Tools run against an **in-memory fake fleet** by default. `OPSGATE_LIVE=1` switches to real EC2; use a throwaway sandbox account.

There is deliberately no default pair of routing models: which Bedrock models you can call depends on your account and region.

## The reusable patterns: `decider_guard`

Copy `src/decider_guard/` into your own project. `client.py` is shared; each pattern is one more file.

### Pattern 1: Guardrails (`gate.py`)

```python
from strands import Agent
from decider_guard import DeciderGate, GatePolicy

agent = Agent(
    tools=[...],
    interventions=[DeciderGate(policy=GatePolicy(skip={"read_only_tool"}))],
)
```

- `DeciderGate`: a Strands `InterventionHandler` returning `Proceed` / `Guide` / `Confirm`.
- `decide(probs, policy, tool, args)`: a pure function from probabilities to a verdict, unit-tested without a model.
- `DEFAULT_QUESTIONS`: the three guardrail questions. Replace them for your domain.
- Fails closed by default: if Decider is unreachable the call is denied. `GatePolicy(fail_closed=False)` fails open.
- A `Confirm` pauses the turn with `stop_reason == "interrupt"`. Resume it with `opsgate.approval.run_with_approval` or your own approval UI.

### Pattern 2: Model routing (`routing.py`)

```python
from decider_guard import ModelRouter

router = ModelRouter(fast_model_id="<cheap id>", frontier_model_id="<strong id>")
decision = router.decide("what's running?")
print(decision.tier, decision.model_id, decision.p_needs_reasoning)
```

- `ModelRouter`: asks one yes/no question, returns a `RouteDecision` (tier, model id, probability, latency).
- `route(p, policy)`: a pure function from probability to tier.
- `RoutingPolicy`: `fast_max=0.30`, `frontier_min=0.70`. Only a confident "no" reaches the fast model. The unsure band and a Decider outage both go to the frontier model, because a wrong cheap answer costs more than extra tokens.
- `router.fast_share()`: the fraction sent to the fast model. Measure it on your traffic before you claim any saving.
- `routing.py` imports no Strands code. `opsgate.routed.RoutedAgent` shows how to apply a decision to a Strands `BedrockModel`.

## Tuning and limits

- **Thresholds are hand-picked.** Tune both policies on your own traffic.
- **Yes/no answers have no confidence field.** The Decider docs' "0.9 confidence is about 95% right" figure applies to choice and score answers. For yes/no answers the probability is the uncertainty, so the policies treat values near 0.5 as "unsure".
- **Questions matter.** The Decider authors say questions are read less than documents and that yes/no tasks transfer poorly from their training mix. Keep questions short and check them against real examples.
- **The routing benchmark is a smoke test.** `bench/routing.py` has ten requests labelled by the author. It is not an accuracy claim.
- Decider 2B is experimental. Use it to veto and escalate, not to replace judgment.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

44 tests run offline. They use a fake Decider HTTP server, so they check our client, state rendering, policies, routing, approval flow and Strands wiring. They do **not** measure Decider's accuracy.

## Deploy on AWS

`deploy/ec2-user-data.sh` starts Decider on one EC2 instance (GPU recommended; CPU works but is slower). `deploy/iam-policy.json` is the minimal Bedrock-only policy. Neither has been run yet.

License: Apache-2.0. See [LICENSE](LICENSE) for details.
