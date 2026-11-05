---
name: trajectory
description: Use the Trajectory SDK to integrate specs (benchmarks), evaluate or train models, inspect run results, and serve checkpoints.
---

# Trajectory

Follow the customer's requested scope. Evaluation alone does not authorize training or
deployment. Preserve the source spec's solving and grading behavior. If credentials
are missing, finish the offline work and state what remains unverified.

## Connect and find the contract

Install `trajectory-sdk`; import `Client` from `trajectory`. It reads `TRAJECTORY_API_KEY`
and `TRAJECTORY_BASE_URL`, or the key `trajectory login` saved for that URL. Confirm the
organization before creating resources. Without credentials, ask the user to run
`trajectory login`, which signs in through the browser. Never ask for an API key in the
conversation, and never print or read the saved credentials file.

Use the project's Python environment for SDK calls and guide discovery (for example,
`uv run python` in a uv project). The installed SDK's signatures and bundled guides are
the API contract; do not assume this skill's installation upgraded that SDK. Print the bundled
docs directory (it holds `agent.mdx`, `platform.mdx`, and `guides/`) with
`python -c "import importlib.metadata as m; print(next(f.locate().resolve().parent for f in m.files('trajectory-sdk') if f.name == 'introduction.mdx'))"`.
The [cookbook](https://github.com/Trajectorylabs/trajectory-cookbook) offers worked examples.

## Link what you report

Make remote work observable without waiting to be asked. When a resource is created,
provide its ID and a labeled platform link; after a long-running action starts, share its
link before waiting. When work finishes, send a completion update with its outcome and
platform link, even if you already shared the link at launch. For failed work, link the
run and relevant failure evidence. In progress and result reports, link the evidence that supports the
claim: the run, selected trajectory, comparison, or deployed model. Prefer one useful
link beside the relevant statement over a repeated list of every resource.

Use `PlatformLinks` from `trajectory.lib.platform_links`; read bundled `platform.mdx`
for its methods. Resolve the organization once with `PlatformLinks.from_client(client)`.
For a nondefault API, pass the supplied or observed matching `platform_url`; never guess
an environment or organization. If the host or required ID is unavailable, report the ID
and that limitation. Never put credentials in URLs.

State what happened, what remains running or unverified, and what the human can inspect
or decide next. Distinguish a submitted job from a successful result. Link a specific
training step or trajectory when its recorded selection is available; otherwise link the
run. Keep local-only findings local; do not invent a platform resource for them.

## Integrate a spec

Read the bundled `agent.mdx` and `guides/spec.mdx`. Adapt examples to the source
spec's tasks, tools, and scoring.
For an existing scorer, Inspect harness, or diagnostic files, read
`guides/source-graders.mdx`, `guides/inspect.mdx`, or `guides/diagnostic-artifacts.mdx`.

Trace the source task loader, actor and tools, grader, and result reader. Preserve their
inputs, call order, state, and primary score. Keep private grader inputs, answers, and
credentials outside actor-visible tools; check what those tools can access.

- Reuse a suitable agent and spec before creating another. Keep source task IDs and
  splits when selecting a subset. Register a revised harness under the same spec
  name and retain its new `bench_id`.
- Route managed model calls to the selected actor. Keep source-fixed judges separate
  from that actor. Use `SecretRef` for runtime credentials.
- Publish the score the source result reader uses. Keep other scores as named components
  when useful; a fallback from an unrun grader is not a grade.
- Report the actual ending. Check `finish_reason` before tool dispatch; a length-capped
  answer is `TRUNCATION` even when the source grades it. A clean process exit alone does
  not establish `ENV_DONE`. Read `guides/termination-reasons.mdx` for other endings.

Save runnable uploader and runtime entrypoints in the customer's repo. Check packaged imports
and representative correct, wrong, cut-off, and error cases. Run a one-task managed
diagnostic or eval, with the model and `options` you will train with, before scaling up;
check each task's ending and reward and the source tools and grader, not only the image
build or aggregate score.

## Evaluate or train

Read the bundled `guides/evaluations.mdx` or `guides/training.mdx` as needed.
Check supported models and options, preserve source generation settings through the run's
`options` (they override request values in managed runs), and budget for grading. Before
training, check eligible endings and the planned sample budget. Watch attempted and graded
rollouts and optimizer progress; stop a validation run that exceeds its planned budget
without progress.

Compare baseline and checkpoint on the same spec version, held-out tasks, scorer, and
effective settings. Report attempted, graded, failed, and unfinished counts before
interpreting means. Check native grades and whether an optimizer update occurred before
attributing a score change to training. A smoke eval does not establish improvement.

## Serve a checkpoint or capture application traffic

Read the bundled `guides/deployments.mdx`.
Verify a completed, nonempty answer, and remove temporary deployments when finished.
Promoting an active production model requires the customer's authorization.
For application traffic, see `guides/production-traces.mdx` (private beta) and retain
the request's trajectory ID.

## Inspect or recover a run

Read the bundled `reference/troubleshooting.mdx`; search it for the error text or symptom.

Use the saved operation or run ID. After a local timeout, retrieve that remote operation
before creating another; stopping a client does not cancel its run. Never complete a
trajectory owned by a live worker.

Locate the failure in the build, runtime, model, harness, grader, reward publication, or
aggregation. Read available logs or artifacts. For flat rewards, distinguish wrong answers
from absent grades, rejected endings, and no training progress.

Report observed evidence with resource IDs and links. Do not guess an unavailable failure
reason. Repeat the affected validation after a fix and confirm requested remote cancellation
or cleanup.

For Trajectory feedback, suggestions, positive observations, or friction, follow the bundled
[agent-feedback skill](../agent-feedback/SKILL.md) when feedback collection is authorized.
It explains what to share and how to submit feedback without blocking the customer's task.
