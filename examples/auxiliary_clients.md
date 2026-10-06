# Use a fixed auxiliary model

A judge or helper can use a fixed Trajectory deployment while the actor uses the model
selected for evaluation or training. Give the helper an organization API key and its own
trajectory. A managed actor credential pins requests to the actor; changing its request
model or trajectory header does not create an independent helper.

Start with an existing committed checkpoint deployment. Use `client.deployments.create(...)`
if you need to create one, then `client.deployments.retrieve(deployment_id)` to check its
status, checkpoint ID, checkpoint step and base model. Wait for `DEPLOYED` before inference.
Record those fields so later evaluations can use the same helper weights. Deployment
readiness alone does not prove a successful inference request.

| Setting | Value |
| --- | --- |
| SDK origin | The API origin, normally `https://api.trajectory.ai` |
| Native OpenAI-compatible base | `SDK_ORIGIN/api/v1/deploy/DEPLOYMENT_ID` |
| Authentication | A separate organization API key; native clients send it as a bearer token |
| Request model | The deployment's `model_slug`, not its `base_model_slug` |
| Trajectory | A new `client.trajectories.create().tid`, passed as `X-Trajectory-Id` |
| Output limit | An explicit per-call `max_tokens`, subject to the model's supported limits |

The exact deployment path works for a deployed test deployment even when `is_active` is
false. Using a production slug through the general inference route follows that slug's
current production target instead. Do not append `/v1` or `/chat/completions` to the native
base: OpenAI and LiteLLM append the completion path themselves. Keep the SDK origin separate
from this native base; trajectory lifecycle calls use the SDK origin.

Run [the complete example](auxiliary_client.py) with either native client:

```bash
pip install --upgrade trajectory-sdk openai litellm
export AUXILIARY_SDK_ORIGIN="https://api.trajectory.ai"
export AUXILIARY_API_KEY="..."  # Organization key, not the managed actor credential.
export AUXILIARY_DEPLOYMENT_ID="dpy_..."
AUXILIARY_CLIENT=openai python examples/auxiliary_client.py
AUXILIARY_CLIENT=litellm python examples/auxiliary_client.py
```

Each invocation makes one small helper request with a 128-token output limit, no automatic
retries and a 600-second request timeout. These are connection-check settings, not changes
to a benchmark's native budgets. A successful response may use its budget on reasoning;
inspect the response and usage rather than requiring literal text from every model.

For a harness, pass the endpoint, key and trajectory header through its supported native
client configuration. Preserve its messages, tools, response format and other generation
settings. Apply your declared helper output ceiling per request, retaining any smaller
native limit; a stored deployment configuration is not a substitute for a request limit.
Avoid changing process-wide actor credentials to configure helpers.

The example completes the helper trajectory on success or error and does not log an actor
reward. Create each helper trajectory with the separate organization client, without a
training run, dataset or task association. Retain its ID alongside the actor's diagnostic
artifacts. Check its public metadata and served deployment, and verify that it is not
attached to the actor rollout or training sample. The actor's reward still comes from the
native scorer; helper errors follow the harness's native failure semantics.

For managed runtimes, store the organization key as a named secret and pass it with
`SecretRef`, as shown in the [Inspect example](inspect.md). Explicitly provide both the key
and SDK origin to the helper's `Client`; the default `Client()` remains the actor client.
Never log credentials or credential-bearing endpoint URLs. See
[public diagnostics](diagnostic_artifacts.md) for reading failed trajectories and reports.
