# Keep diagnostic files with a trajectory

Use `log_event` for structured summaries. Store full test logs, reports, or harness transcripts
as artifacts, then include the artifact ID in an event. Events and rewards are separate: record
the native grader's score with `log_reward`.

The serialized event name and payload must fit within 10 MiB. An artifact can contain up to
16 MiB; compress larger text reports or split them into files before uploading.

With an active trajectory ID `tid`, this example uploads a compressed report without discarding
its contents:

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
