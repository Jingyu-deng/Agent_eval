# Agent Eval

A customer-support **refund agent** built with LangChain / LangGraph, together with a
**DeepEval** task-completion evaluation suite that scores the agent against a set of
scripted customer conversations.

## What the agent does

The agent handles refund requests end to end. It identifies the customer, looks up their
purchase history, checks eligibility, and either issues a refund ticket or explains why
the refund is denied.

Policy rules it enforces:

- No refund more than **30 days** after purchase, and no purchase dated in the future.
- Only the items the customer actually asked to return are refunded.
- Only items the customer actually purchased are refundable.
- Unknown usernames are rejected, and the customer is asked for a valid one.

The agent returns a structured `Answer` (`answer: str`, `conversation: bool`). When
`conversation` is `False`, the decision has been made and the CLI loop ends.

### Tools

| Tool | Purpose |
|---|---|
| `get_user_purchase_history` | Look up a user's purchases and purchase dates |
| `calculate_right_refund` | Compute a refund, enforcing the 30-day / purchased-item rules |
| `issue_refund_ticket` | Issue the refund (returns a confirmation message) |
| `get_today_date` | Return today's date in `YYYY-MM-DD` |

## Project layout

```
agent/                 The refund agent
  agent.py             CLI loop + agent construction (LangGraph checkpointer)
  models.py            Chat model (DeepSeek)
  prompts.py           System prompt with the refund guidelines
  objects.py           Structured response models (Answer, PurchaseItem)
  tools.py             Tool implementations
  data.py              Sample user/purchase data
evaluations/                DeepEval evaluation suite
  dataset.py                7 golden test cases with expected outputs/tool calls
  model.py                  Judge model (DeepSeek via DeepEval; loads .env)
  eval_all.py               Runs all seven metrics in one go
  task_completion.py        TaskCompletionMetric
  answer_correctness.py     GEval answer correctness
  argument_correctness.py   ArgumentCorrectnessMetric
  tool_correctness.py       ToolCorrectnessMetric
  plan_adherence.py         PlanAdherenceMetric
  plan_quality.py           PlanQualityMetric
  step_efficiency.py        StepEfficiencyMetric
src/agent_eval/             Python package stub
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

Then put your DeepSeek API key in `.env`:

```
DEEPSEEK_API_KEY="sk-..."
```

`.env` is git-ignored — never commit it.

## Run the agent

The agent modules use flat imports, so run from the `agent/` directory:

```bash
cd agent
uv run python agent.py
```

## Run the evaluations

Each script in `evaluations/` runs the agent against the 7 goldens and scores one
metric, printing a per-test-case table and an aggregate report. Run them from the
`evaluations/` directory (the scripts use flat imports):

```bash
cd evaluations

uv run python task_completion.py      # TaskCompletionMetric
uv run python argument_correctness.py # ArgumentCorrectnessMetric
uv run python tool_correctness.py     # ToolCorrectnessMetric
uv run python answer_correctness.py   # GEval answer correctness
uv run python plan_adherence.py       # PlanAdherenceMetric
uv run python plan_quality.py         # PlanQualityMetric
uv run python step_efficiency.py      # StepEfficiencyMetric
```

`eval_all.py` runs all seven metrics against a single set of agent responses:

```bash
uv run python eval_all.py
```

The agentic metrics (`TaskCompletionMetric`, `PlanQualityMetric`, `PlanAdherenceMetric`,
`StepEfficiencyMetric`) need the LangGraph trace captured by DeepEval's
`CallbackHandler`; the others score the structured `Answer` directly.

Notes:

- The judge model is `deepseek-v4-flash`, set in `evaluations/model.py`. The file loads
  `.env` before constructing the model, so the scripts work from any directory.
- Evaluation runs cost API credits — each golden is a full agent conversation plus
  several judge calls.
- Concurrency is capped at 4 (`AsyncConfig(max_concurrent=4)`) because this DeepSeek
  account allows 5 concurrent requests; DeepEval's default of 20 triggers HTTP 429.
- Run artifacts are written to `evaluations/.deepeval/` and are git-ignored.

## Notes

- **Windows, GBK consoles:** DeepEval's report contains emoji. If you see
  `UnicodeEncodeError: 'gbk' codec can't encode character ...`, run with UTF-8 output:
  `$env:PYTHONUTF8=1` (PowerShell) or `PYTHONUTF8=1` (cmd/bash) before the command.
- **Line endings:** `.gitattributes` normalizes all text files to LF, overriding
  `core.autocrlf` on Windows.
- The checkpointer is in-memory only (`InMemorySaver`), so conversations are lost when
  the process exits.
