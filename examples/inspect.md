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

Report the native scorer's value, including zero. If it produces labels or several
components, use the benchmark's defined conversion or report the components separately.
Do not substitute zero for an execution error or missing score.

Pass these transport options through any wrapper around `eval_async`. Keep the native
solver and tool implementations, prompts, stopping conditions, and model settings.
A managed Trajectory run selects the actor endpoint; keep the native model identifier
where the harness uses it to configure behavior.

The options above affect this Inspect actor client. Leave auxiliary clients, such as
an LLM judge or a question-answering tool, configured as the benchmark specifies. Avoid
changing process-wide `OPENAI_API_KEY` or `OPENAI_BASE_URL` to redirect the actor: those
variables may also configure auxiliary clients.
