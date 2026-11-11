# BigFinanceBench

> [!TIP]
> **Run this example with a coding agent**
>
> 1. In the [Trajectory platform](https://platform.trajectory.ai), click **Copy setup prompt**.
> 2. Paste the prompt into your coding agent of choice (Claude Code, Codex, Vercel, etc.).
> 3. Then paste this example's GitHub link and ask the agent to run it:
>
>    ```text
>    https://github.com/Trajectorylabs/trajectory-cookbook/tree/main/examples/big_finance_bench
>    ```

<img width="1056" height="425" alt="copy-setup-prompt" src="https://github.com/user-attachments/assets/c5c03c36-e07d-4789-a05e-5aa4a59159a8" />


## Dataset

[BigFinanceBench](https://github.com/Rogo-Technologies/big-finance-benchmark) from Rogo
asks financial-research questions graded against expert, point-weighted rubrics. This
example uploads the 50-question public subset at commit
`3b13eecc4cacb4cfa13ac2b7c111d14ce127dd71`. The other 878 questions are held back upstream.

## Harness

The upstream ReAct agent, system prompt, and five tools (`web_search`, `edgar_search`,
`fetch_url`, `python_exec`, `final_answer`) run unchanged except for one isolation patch
(see the Appendix). A small adapter sends the agent's model calls to Trajectory, grades the
run with the upstream rubric judge, and logs `rubric_points`: points earned out of points
possible. The explanation also records whether the final answer matched the reference, and
the `big_finance_bench_grade` event keeps each rubric line's verdict.

## Ingestion

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), Git, and a Trajectory organization
API key. Register three organization secrets: `OPENAI_API_KEY` for the judge, `TAVILY_API_KEY`
for web search, and `SEC_EDGAR_USER_AGENT` (a name and email, as SEC EDGAR requires).

```sh
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.10.1' python -c '
from trajectory import Client
client = Client()
client.secrets.create(name="OPENAI_API_KEY", value="YOUR_OPENAI_API_KEY")
client.secrets.create(name="TAVILY_API_KEY", value="YOUR_TAVILY_API_KEY")
client.secrets.create(name="SEC_EDGAR_USER_AGENT", value="Your Name you@example.com")
client.agents.create(name="big-finance-bench", description="BigFinanceBench evaluation")'
uv run examples/big_finance_bench/ingest.py --agent-name big-finance-bench \
  --task bf-6bd9e9e383 --task bf-c75541fef6
```

The command fetches the pinned upstream source, uploads the two selected questions, waits
for the image, and runs a diagnostic with Qwen 3.6 35B A3B. It prints the benchmark and
diagnostic IDs. A reward of zero is a valid grade; a missing grade or failed task is not.
Omit `--task` to upload all 50 questions. To grade through OpenRouter, register
`OPENROUTER_API_KEY` instead of `OPENAI_API_KEY` and add `--judge-model openrouter/openai/gpt-5.5`.

## Evaluation

To evaluate the same uploaded version, use the printed `bench_id`:

```python
from trajectory import Client

run = Client().evals.create(
    bench_id="YOUR_BENCH_ID",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "reasoning_effort": "on",
        "max_turns_per_trajectory": 50,
        "max_output_tokens_per_turn": 65536,
        "max_output_tokens_per_trajectory": 262144,
        "trajectory_timeout_seconds": 3600,
    },
)
print(run.eval_run_id)
```

The turn and output limits match upstream's 50-step and 65,536-token defaults. Each
question is graded once by `openai:gpt-5.5`, the judge in upstream's quickstart; the
paper's headline numbers average a two-judge panel over three trials.

## Training

Upstream marks every question `evaluation_only` and `do_not_train`, so this example uploads
evaluation tasks only.

## Appendix

The code is [Apache-2.0](https://github.com/Rogo-Technologies/big-finance-benchmark/blob/3b13eecc4cacb4cfa13ac2b7c111d14ce127dd71/LICENSE)
and the questions are [CC BY 4.0](https://github.com/Rogo-Technologies/big-finance-benchmark/blob/3b13eecc4cacb4cfa13ac2b7c111d14ce127dd71/data/LICENSE-DATA),
© Rogo Technologies; the image keeps both licenses and the citation file. Cite Wang et al.,
“BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents”
([arXiv:2606.03829](https://arxiv.org/abs/2606.03829)).

Upstream's `python_exec` runs model-written code in a child process that inherits the
harness's credentials and can read the bundled answers and rubrics. The runtime
[patch](runtime/python_exec.patch) runs that child as `nobody` with an empty environment,
and the dataset is readable only by root, which runs the harness and grader. Timeouts,
output limits, and the tool description are unchanged.
