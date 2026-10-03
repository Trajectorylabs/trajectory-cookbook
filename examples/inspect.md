# Connect an Inspect harness

If your benchmark uses [Inspect](https://inspect.aisi.org.uk/), keep its task, solver,
sandbox, scorer, and generation settings. Configure the actor's client at the existing
`eval_async` call using `model_base_url` and `model_args`.

This pattern applies to Inspect's OpenAI provider and OpenAI-compatible clients. Other
provider protocols need a compatible adapter; changing the model's provider prefix can
also change how the harness interprets its configuration.

At the call below, `task` is your existing Inspect task with one sample, `model` is its
configured model, and `eval_options` contains its existing evaluation settings. Replace
`accuracy` with your scorer's name. If you already supply `model_args`, preserve its
non-transport options when adding the credentials and header.

```python
from inspect_ai import eval_async
from trajectory import Client

client = Client()
tid = client.trajectories.create().tid
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
reward = float(samples[0].scores["accuracy"].value)
client.trajectories.log_reward(
    tid, reward_id="accuracy", name="reward_accuracy", value=reward,
)
client.trajectories.complete(tid)
```

Report the native scorer's value, including zero. For labels or multiple components,
preserve the benchmark's defined conversion and weighting. Logged reward components are
summed with their weights; record diagnostic scores using `client.trajectories.log_event(...)`
or `client.trajectories.log_reward(..., weight=0)`. Do not substitute zero for an execution
error or missing score. Preserve the native scorer's acceptance rules; report suspected
grading defects separately rather than adding new pass/fail conditions during integration.

Pass these transport options through any wrapper around `eval_async`. Keep the native
solver and tool implementations, prompts, stopping conditions, and model settings.
A managed Trajectory run selects the actor endpoint; keep the native model identifier
where the harness uses it to configure behavior. Requests through that managed session use
its actor endpoint even when `model` names another model. Give auxiliary judges and tools
their own provider clients and credentials.

The options above affect this Inspect actor client. Preserve the effective configuration
of auxiliary clients, such as an LLM judge or a question-answering tool. Resolve their
models and settings through the native entrypoint, including any defaults inherited from
the actor model. Optional configuration examples in a README are not default settings.
Avoid changing process-wide `OPENAI_API_KEY` or `OPENAI_BASE_URL` to redirect the actor:
those variables may also configure auxiliary clients.

Use provider credentials that remain valid when the task runs. A temporary model proxy
used by the agent preparing the dataset may expire with that agent's session or allow
only its model and API routes; do not store that proxy token as a runtime provider key.

Auxiliary clients need credentials for their provider. A compatible endpoint can supply
the same model without changing the native tool. For example, to use an Anthropic client
through [OpenRouter](https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-messages),
register your OpenRouter key as an organization secret and add these entries to the task's
`env_vars`:

```python
from trajectory import SecretRef

auxiliary_env = {
    "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
    "ANTHROPIC_API_KEY": SecretRef(secret_ref="OPENROUTER_API_KEY"),
}
```

Keep the benchmark's auxiliary model identifier and generation settings. Verify that the
endpoint supports that model and its required features. These variables configure Anthropic
clients; keep actor routing explicit as above.
