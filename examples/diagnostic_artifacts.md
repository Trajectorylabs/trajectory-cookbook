# Keep diagnostic files with a trajectory

Use `log_event` for structured summaries and artifacts for test logs, reports, or harness
transcripts. Include the artifact ID in an event so it can be found alongside the task's other
records. If a diagnostic upload fails, report the upload error separately from the task's
execution result and computed reward.

The serialized event name and payload must fit within 10 MiB. An artifact can contain up to
16 MiB; compress larger text reports or split them into files before uploading.

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
