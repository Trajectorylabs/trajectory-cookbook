# Keep diagnostic files with a trajectory

## Find diagnostics for a failed evaluation

Use `trajectory-sdk>=0.8.13` to list trajectories by evaluation ID, including unfinished
and failed attempts. A failed attempt may have no reward, so a reward listing is not a
complete list of the run's trajectories.

```python
from trajectory import Client

client = Client()
run_id = "evr_YOUR_RUN"
run = client.evals.runs.retrieve(run_id)
print(run.status, run.failure)

for trajectory in client.trajectories.list(eval_run_id=run_id):
    print(trajectory.trajectory_id, trajectory.status)
    for event in client.trajectories.list_events(trajectory.trajectory_id):
        print(event.event_id, event.event)
```

Both listings paginate when iterated. If an event records an artifact ID, retrieve the
artifact as described below. Inspect the native error or report before repairing and
retrying the affected task. An absent reward remains ungraded; it is not a zero score.

## Record events and artifacts

`log_event` records a named JSON payload on a trajectory without changing its reward. Use a
different `event_id` for each distinct event and reuse that ID only when retrying the same
write. An existing ID is deduplicated, not updated. Record new events before completing the
trajectory; read them later with `client.trajectories.list_events(tid)`.

Use events for structured diagnostics and artifacts for files such as test logs or reports.
Completing an artifact upload attaches the file to the trajectory. The event in this example
also records its ID and filename. If a diagnostic upload fails, report it separately from the
task's execution result and computed reward.

An artifact upload can contain up to 16 MiB. Compress larger text reports or split them into
files before uploading.

With an active trajectory ID `tid`, upload a compressed report:

```python
import base64
import gzip
import hashlib
from pathlib import Path

import httpx
from trajectory import Client

client = Client()
content = gzip.compress(Path("report.json").read_bytes(), mtime=0)
upload = client.trajectories.artifacts.create_upload(
    tid,
    media_type="application/gzip",
    size_bytes=len(content),
    md5=base64.b64encode(hashlib.md5(content).digest()).decode(),
)
response = httpx.put(upload.upload_url, content=content, headers=upload.headers, timeout=120)
response.raise_for_status()
artifact = client.trajectories.artifacts.complete_upload(tid, upload.artifact_id)
client.trajectories.log_event(
    tid,
    event_id="diagnostic-report",
    name="diagnostic_report",
    payload={"artifact_id": artifact.artifact_id, "filename": "report.json.gz"},
)
```

Finish the upload before completing the trajectory. To read the file later, call
`client.artifacts.retrieve(artifact.artifact_id)` and download its `download_url`.
