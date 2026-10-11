# Keep diagnostic files with a trajectory

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
from urllib.request import Request, urlopen

from trajectory import Client

client = Client()
content = gzip.compress(Path("report.json").read_bytes(), mtime=0)
upload = client.trajectories.artifacts.create_upload(
    tid,
    media_type="application/gzip",
    size_bytes=len(content),
    md5=base64.b64encode(hashlib.md5(content).digest()).decode(),
)
request = Request(upload.upload_url, data=content, headers=upload.headers, method="PUT")
with urlopen(request, timeout=120) as response:
    response.read()
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

For score explanations, optional labeled evidence and the customer UI/SDK feedback loop, see
[evaluation inspection and repair](evaluation_inspection.md). Its additional inspection
interfaces require the matching release described there.
