---
name: agent-feedback
description: Send customer-authorized Trajectory feedback, suggestions, or positive observations as prose or partial structured details about the API, SDK, documentation, or workflow.
---

# Agent feedback

Send actionable product feedback through Trajectory's authenticated feedback API. Reports
go to the Trajectory team's Slack inbox. This is separate from scoring an agent's answers
or recording trajectory rewards.

## Choose useful evidence

Share observations from using Trajectory: what helped, a suggestion, confusing documentation,
a missing capability, a bug, or friction you worked around. Separate what you observed
from a suggestion or suspected cause. Do not invent failures or classify a task's ordinary
wrong answer as a platform bug.

If the customer corrected your approach or stated a preference, identify that as customer
feedback and describe the correction when sharing it is authorized. Label your own inferred
expectation as a suggestion. Preserve the distinction so reviewers can judge the evidence.

Follow the customer's instructions about sharing feedback. A setup prompt asking you to
report Trajectory feedback authorizes that reporting; otherwise obtain permission before
sending. Submit one concise report per distinct issue in the session, not one per retry.
Continue the requested task when possible; reporting must not become a prerequisite.

## Prepare the report

Start with a short prose message. No category or structured details are required:

```json
{
  "message": "The image-build guide did not explain how to check readiness. I found benchmarks.retrieve_readiness in a second guide and continued the evaluation."
}
```

Add only the details that help explain the experience; do not fill fields just to complete
a form. You can also send a partial structured report without `message`.
The API enforces these character and item limits before Slack delivery; invalid reports
return `422`.

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

Include a minimal reproduction when it clarifies the issue. Use existing request or
resource IDs to correlate it. Exclude API keys, authorization headers, signed URLs,
private prompts, customer source code, and full conversation or log dumps. Do not copy the
setup prompt into the report; it may contain a key.

## Submit once and report the result

Use `TRAJECTORY_API_KEY` and the same trusted `TRAJECTORY_BASE_URL` as the customer's
Trajectory client (default `https://api.trajectory.ai`). Do not obtain Slack credentials
or send directly to Slack. Save the reviewed payload as `trajectory-feedback.json`, then:

```bash
feedback_api="${TRAJECTORY_BASE_URL:-https://api.trajectory.ai}"
curl --fail-with-body --silent --show-error --max-time 10 --retry 0 \
  "${feedback_api%/}/api/v1/feedback" \
  -H "X-API-Key: ${TRAJECTORY_API_KEY:?TRAJECTORY_API_KEY is required}" \
  -H 'Content-Type: application/json' \
  --data-binary @trajectory-feedback.json
```

Keep shell tracing and verbose HTTP logging disabled. The API returns `201` and an
`id` receipt only after Slack accepts the report. Say it was submitted only after that
response. Each organization has 10 submission attempts per one-hour window, including
failed deliveries. If rate limited, stop reporting for this task. A timeout can leave
delivery uncertain: do not automatically resubmit, because
the endpoint does not promise deduplication. If reporting fails, state that briefly and
continue the customer's task; do not submit a second report about the feedback failure.
