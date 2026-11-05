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
and `TRAJECTORY_BASE_URL`. Confirm the organization before creating resources.

Use the installed SDK's signatures and bundled guides as the contract. Print the bundled
docs directory (it holds `agent.mdx`, `platform.mdx`, and `guides/`) with
`python -c "import importlib.metadata as m; print(next(f.locate().resolve().parent for f in m.files('trajectory-sdk') if f.name == 'introduction.mdx'))"`.
The [cookbook](https://github.com/Trajectorylabs/trajectory-cookbook) offers worked examples.

## Link what you report

When you report a spec, evaluation, training run, or trajectory (its creation, progress,
result, or failure) or name one in a next step, link its platform page beside its ID. Build
the link from the installed `platform.mdx` with the `organization_key` from
`client.organizations.retrieve_current()`. A trajectory without its own path there, such as
a training attempt, links to its run's page. For a `TRAJECTORY_BASE_URL` other than the
default, use only a given or observed platform host; without one, report IDs without links.

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
