# Connect an Inspect harness

For an [Inspect](https://inspect.aisi.org.uk/) harness using the OpenAI provider, configure
its actor client at the `eval_async` call using `model_base_url` and `model_args`. Keep the
task, solver, sandbox and scorer. Other provider protocols require a compatible adapter;
changing the provider prefix can also change how Inspect interprets model settings.

Here, `task` contains one sample, `model` is the harness's configured model, and `eval_options`
contains its evaluation settings. Replace `accuracy` with your scorer's name. The example uses
Inspect's standard scalar conversion (`C` → 1, `I` → 0, `P` → 0.5, `N` → 0); use your
benchmark's conversion for custom labels or multiple components. If you already pass
`model_args`, merge the credentials and headers below with its existing options.

```python
import logging
import math

from inspect_ai import eval_async
from inspect_ai.scorer import value_to_float
from trajectory import Client

client = Client()
tid = client.trajectories.create().tid
try:
    logs = await eval_async(
        task,
        model=model,
        model_base_url=f"{str(client.base_url).rstrip('/')}/v1",
        model_args={
            "api_key": client.api_key,
            "default_headers": {"X-Trajectory-Id": tid},
        },
        **eval_options,
    )
    if len(logs) != 1 or logs[0].status != "success":
        raise RuntimeError("Inspect did not complete successfully")
    samples = logs[0].samples
    if samples is None or len(samples) != 1 or samples[0].error is not None:
        raise RuntimeError("Expected one completed Inspect sample")
    value = samples[0].scores["accuracy"].value
    if not isinstance(value, (str, int, float)):
        raise ValueError("Use the benchmark's conversion for structured scores")
    if (
        isinstance(value, str)
        and value not in {"C", "I", "P", "N"}
        and value.lower() not in {"yes", "no", "true", "false"}
    ):
        value = float(value)  # Reject unknown labels instead of silently recording zero.
    reward = value_to_float()(value)
    if not math.isfinite(reward):
        raise ValueError("Inspect did not produce a finite score")
    client.trajectories.log_reward(
        tid, reward_id="accuracy", name="reward_accuracy", value=reward,
    )
    client.trajectories.complete(tid)
except Exception:
    try:
        client.trajectories.complete(tid, termination_reason="ERROR")
    except Exception:
        logging.exception("Failed to report trajectory failure")
    raise
```

The example reports genuine zero scores and marks missing scores or execution failures as
errors. See [reward and completion semantics](../README.md#report-native-results) for component
weights and termination reasons.

A managed Trajectory run selects the actor model even when the request names another model.
Keep the harness's model identifier where it affects prompts or other behavior. Pass the
transport options above through any wrapper around `eval_async`.

Configure auxiliary clients, such as an LLM judge, separately so their requests reach the
intended endpoint. Avoid changing process-wide `OPENAI_API_KEY` or `OPENAI_BASE_URL` to redirect
the actor if auxiliary clients read those variables. A judge that inherits settings from the
actor's model name may need explicit configuration when the managed actor differs.

Use organization secrets for credentials the runtime needs. The following example is for a
harness with an Anthropic judge; use the secret names and provider required by your harness.
The credentials must remain valid when the task runs, beyond any temporary ingestion-agent
session.

```python
from trajectory import SecretRef

auxiliary_env = {
    "ANTHROPIC_API_KEY": SecretRef(secret_ref="ANTHROPIC_API_KEY"),
}
```

Create the named secret with `client.secrets.create(...)` before execution and pass
`auxiliary_env` as the task's `env_vars`.

If you implement an Inspect provider adapter, [register it with `@modelapi`](https://inspect.aisi.org.uk/extensions-model-api.html)
and obtain it through `get_model(...)`. Its `should_retry` hook should honor
`x-should-retry: false` on an `APIStatusError`'s response headers, since Inspect can retry
errors after the underlying client has stopped. Otherwise retain the provider's retry policy.
