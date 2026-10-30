# Big Finance Benchmark

This recipe adapts an existing ReAct research benchmark while preserving its agent loop, web and
SEC tools, and rubric grader. It packages a 50-question subset; the rest of the benchmark stays
held back.

Use this pattern when a benchmark already has an OpenAI-compatible model abstraction and its own
orchestrator. The SDK adapter connects the existing lifecycle to one Trajectory session:

1. Create one trajectory when a question starts.
2. Forward the solving agent's model requests with the same trajectory ID.
3. Run the benchmark's original rubric grader.
4. Record the normalized rubric score and complete the trajectory.

## Adapt the existing model client

The adapter implements the benchmark's existing `ModelClient` interface. It converts the
existing messages and tools, then forwards each request through the Trajectory session model:

```python
self.client = Client()
self.tid = self.client.trajectories.create().tid

response = await asyncio.to_thread(
    self.client.chat.completions.create,
    model="trajectory-session",
    messages=_to_oai_messages(system, messages),
    x_trajectory_id=self.tid,
    extra_body={"tools": _to_oai_tools(tools), "tool_choice": "auto"},
)
```

The adapter maps that response back into the benchmark's native response type, so its ReAct loop
and tool execution remain unchanged.

## Reuse the original grader

After the agent finishes, the orchestrator calls the existing rubric grader and normalizes its
score. The adapter's trajectory receives the reward, then completes with the reason the agent
stopped. A harness error or grader failure completes `ERROR` without a reward; a context overflow
is still graded, as the native harness does, and completes `TRUNCATION`:

```python
client, tid = adapter.client, adapter.tid
try:
    if run.stop_reason == "error":
        raise RuntimeError(f"agent run failed ({run.stop_reason}): {run.error}")
    graded = asyncio.run(grade(run=run, item=item, judge_model_id=judges[0]))
    earned, possible = graded.rubric_points_earned, graded.rubric_points_possible
    client.trajectories.log_reward(
        tid, name="reward_accuracy", value=earned / possible,
        explanation=f"{earned}/{possible} rubric points",
    )
except Exception:
    client.trajectories.complete(tid, termination_reason="ERROR")
    raise
endings = {
    "max_steps": "MAX_STEPS",
    "context_exceeded": "TRUNCATION",
    "token_budget": "TRUNCATION",
}
client.trajectories.complete(tid, termination_reason=endings.get(run.stop_reason, "ENV_DONE"))
```

## Package the evaluation set

The uploader creates one evaluation task for each subset question. `BFB_TASK_ID` selects exactly
one row inside the existing orchestrator, while credentials remain platform secret references:

```python
from trajectory import SecretRef, TaskSpec
from trajectory.types.benchmarks.task_spec import EnvResources

TaskSpec(
    name=f"big-finance/{row['id']}",
    split="test",
    run_command=(
        "python -u /opt/big_finance/scripts/run_eval_set.py --trajectory "
        "--dataset /opt/big_finance/data/big_finance_subset.jsonl "
        "--run-id trajectory --max-steps 30 --judge openai:gpt-5.4-mini"
    ),
    env_vars={
        "BFB_TASK_ID": row["id"],
        "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
        "TAVILY_API_KEY": SecretRef(secret_ref="TAVILY_API_KEY"),
        "SEC_EDGAR_USER_AGENT": SecretRef(secret_ref="SEC_EDGAR_USER_AGENT"),
    },
    env_resources=EnvResources(network_mode="public"),
)
```

The Docker build uses an allowlisted context containing only the runtime code and the subset.
The held-back benchmark data is never read, uploaded, or copied into the image.
