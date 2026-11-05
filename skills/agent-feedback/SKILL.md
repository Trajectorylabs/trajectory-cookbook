---
name: agent-feedback
description: Send customer-authorized Trajectory feedback, suggestions, or positive observations as prose or partial structured details about the API, SDK, documentation, or workflow.
---

# Agent feedback

Send actionable product feedback through Trajectory's authenticated feedback API. Reports
go to the Trajectory team's Slack inbox. This is separate from scoring an agent's answers
or recording trajectory rewards.

## Choose useful evidence

Report useful observations, suggestions, bugs, or friction from using Trajectory.
Distinguish observed behavior, customer feedback, and your own suggestions. An agent's
wrong answer alone is not a platform bug.

Follow the customer's sharing instructions. A setup prompt requesting Trajectory feedback
authorizes it; otherwise ask before sending. Send one report per distinct issue, not per
retry. Reporting must not block the customer's task.

## Prepare the report

Start with a short prose message. No category or structured details are required:

```json
{
  "message": "The image-build guide did not explain how to check readiness. I found benchmarks.retrieve_readiness in a second guide and continued the evaluation."
}
```

Include only useful details. Partial structured reports can omit `message`. The API
validates these limits before Slack delivery; invalid reports return `422`.

| Optional field | Content and limit |
| --- | --- |
| `message` | General prose, up to 2,000 characters. |
| `summary` | A short description, up to 200 characters. |
| `goal`, `expected_behavior`, `actual_behavior`, `workaround` | Relevant context, up to 1,000 characters each. |
| `category` | Any useful label up to 64 characters; `bug`, `docs`, `feature_request`, and `friction` are suggestions. |
| `operation` | The relevant API or SDK operation, up to 200 characters. |
| `outcome` | Any useful outcome up to 200 characters; examples are `blocked`, `completed_with_workaround`, and `completed`. |
| `sdk_version` | Installed SDK version, up to 100 characters. |
| `resource_ids`, `request_ids` | Up to 10 relevant IDs per list; each ID must be nonblank and 1–200 characters. |

Provide at least one nonblank `message`, `summary`, `expected_behavior`, `actual_behavior`,
`workaround`, or `goal`. Blank fields are ignored; labels, IDs, and version metadata alone
do not constitute a report. Obtain the installed version, when useful, with
`importlib.metadata.version("trajectory-sdk")`.

Include a minimal reproduction and existing request or resource IDs when useful. Exclude API keys, authorization headers, signed URLs,
private prompts, customer source code, and full conversation or log dumps. Do not copy the
setup prompt into the report; it may contain a key.

## Submit once and report the result

Use the same Python environment and trusted API base URL as the customer's SDK calls.
`Client()` uses `TRAJECTORY_API_KEY` or the key saved by `trajectory login`; do not read
credential files yourself. Save the reviewed payload as `trajectory-feedback.json`, then:

```python
import json
from pathlib import Path

import httpx2
from trajectory import Client

payload = json.loads(Path("trajectory-feedback.json").read_text())
with Client(max_retries=0, timeout=10, http_client=httpx2.Client(follow_redirects=False)) as client:
    receipt = client.post("/api/v1/feedback", body=payload, cast_to=dict)
    print("Feedback receipt:", receipt["id"])
```

Keep shell tracing and verbose HTTP logging disabled. The API returns `201` and an
`id` receipt only after Slack accepts the report. Say it was submitted only after that
response. Each organization has 10 submission attempts per one-hour window, including
failed deliveries. If rate limited, stop reporting for this task. A timeout can leave
delivery uncertain: do not automatically resubmit, because
the endpoint does not promise deduplication. If reporting fails, state that briefly and
continue the customer's task; do not submit a second report about the feedback failure.
