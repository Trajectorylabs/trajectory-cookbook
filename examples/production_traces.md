# Collect production traces

Private beta. Record calls from an application under an existing agent. For repeatable
evaluation or training with runnable tasks and a grader, use the
[benchmark workflow](gsm8k/README.md).

Create a trajectory for each application request:

```python
from trajectory import Client

client = Client()
session = client.trajectories.create(
    body={"agent_id": "<agent-id>"},
    idempotency_key="<stable-application-request-id>",
)
tid = session.tid
```

Pass `x_trajectory_id=tid` to each model call, as in
[the capture example](../README.md#record-a-task). Reuse the request's
idempotency key when retrying creation. The organization API key authenticates these calls.

When a grader is available, [log its reward](reward_integration.md). Complete the trajectory
with the [reason the application actually stopped](termination_reasons.md); a model or tool
error is not a normal completion or a graded zero. Without a grader, complete without a reward.

Find these trajectories in the agent's **Production traces** page. Keep the returned `tid` with the
application request so you can retrieve it with `client.trajectories.retrieve(tid)`.

## Record later application events

Use `trajectories.log_event` while the trajectory is running. For later events, use the
organization client and retain the application's session and trajectory IDs:

```python
from datetime import datetime, timezone

result = client.telemetry.ingest_events(events=[{
    "event_id": "request-example-001:answer-accepted",
    "event_type": "answer.accepted",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "session_id": "session-example-001",
    "source": "support-app",
    "trajectory_id": tid,
    "properties": {"surface": "chat"},
}])
print(result.ingested, result.skipped, result.failures)
```

Reuse an event ID when retrying the same event; accepted duplicates count as ingested.
Check `failures` as well as the counts. An application event records what happened; it does
not publish a reward. Use [reward integration](reward_integration.md) for grading.
