---
name: trajectory
description: Integrate a customer benchmark with Trajectory, evaluate and train models, inspect failures and compare results, and deploy trained checkpoints using the Trajectory SDK and cookbook.
---

# Trajectory

Use Trajectory for the customer's requested workflow; evaluation alone does not authorize
training or deployment. Preserve the benchmark's solving and grading behavior. If live
credentials are missing, finish offline validation and state what remains unverified.

## Find the contract

The Python package is `trajectory-sdk`; import `Client` from `trajectory`. It reads
`TRAJECTORY_API_KEY` and `TRAJECTORY_BASE_URL` (default `https://api.trajectory.ai`).
Authentication uses `X-API-Key`. Check the selected organization before creating resources.

Start with the public [cookbook](https://github.com/Trajectorylabs/trajectory-cookbook).
Its [GSM8K walkthrough](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md)
covers the managed lifecycle. Adapt the workflow, not GSM8K's task, model choice, or scoring assumptions.
Use the installed SDK's signatures and docstrings to check calls and options; record its
version. Cookbook examples pin a version. If a method or field differs, resolve the mismatch
before changing the integration.
Do not depend on access to a login-gated documentation site.
If a cookbook snapshot is supplied with the customer source, use that version for the run.

## Integrate and validate

Trace the customer's task loader, actor interface, model calls (if present), grader, and
result reader. If the source has no agent, implement a basic model actor through its documented
interface. Call the source grader directly; preserve its call order and
shared state. Identify task IDs, splits, runtime packages and data, network access, private
references, and the primary score.
Preserve sampling settings, output limit, and stop rules when comparing scores. Save the
uploader, dispatcher, and runtime build files in the customer repo; validate those saved
entrypoints. Place the dispatcher around the source's actor and verifier boundary; the runtime
must execute the harness and publish its outcome. Keep judge and organization keys and private
grader files in the trusted harness, outside the actor tool's environment.

Before upload, check the SDK's model-request signature. Run a build-time import check for the
packaged harness and grader. Test representative source IDs from each split and grader path,
including their runtime data. Cover correct, wrong, cut-off, and exceptional outcomes. Local
Docker is optional: use a managed one-task diagnostic to check execution, grading, and
artifacts.
An import check alone is insufficient. Probe actor-facing tools with a fake secret and
test-only private file.

Once the image is ready, inspect registered task splits, network mode, and secrets. Check one
managed task's raw grade, effective reward, ending, and visible answer before expanding.

- [Choose or reuse an agent](https://github.com/Trajectorylabs/trajectory-cookbook#choose-or-create-an-agent)
  for the customer's project. Inspect existing work before creating another benchmark or run.
- Follow [runtime packaging](https://github.com/Trajectorylabs/trajectory-cookbook#package-a-benchmark-runtime)
  for build contexts, dependencies, secrets, nested containers, and actor versus judge routing.
  Preserve required paths, inputs, grading rules, and the actor's tool interface. For nested
  Docker, put `RUN dockerd --version && docker --version` in its Dockerfile; the managed image
  build checks those binaries, and sandbox readiness runs `docker info`. Set the source's
  network mode explicitly; task networking defaults to none. The managed client routes model
  calls to the actor even when they name another model. A fixed judge needs a separate client and
  secret; changing its model leaves benchmark score parity unverified. The injected
  `TRAJECTORY_API_KEY` is an actor-session key; for auxiliary Trajectory calls, use an
  organization secret and the runtime's injected base URL. Follow the
  [separate-client example](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/auxiliary_inference.py)
  and check eval or training model availability before packaging.
- Follow [reward integration](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/reward_integration.md)
  to map the score the source result reader actually consumes. Validate wrong, partial, and
  correct outputs where applicable. Report native grades from the raw component, not
  `trajectory.reward`. A missing grade or execution failure is not a zero.
- Report [how the attempt ended](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/termination_reasons.md).
  Read the model's `finish_reason`; a clean harness exit or zero grade does not make a
  cut-off reply normal. Preserve native grades for cut-off replies and mark `TRUNCATION`.
  Preserve retry headers; report unexpected harness errors as `ERROR` and re-raise.
- A [task diagnostic](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/task_diagnostics.md)
  can run the packaged entrypoint before bulk upload. Where available, use known-good and
  known-bad solutions to check the adapter independently of model quality.

Keep source task identities and explicit train/test membership. Account for omitted or failed
tasks. If the source has no split metadata, label any partition you make as provisional; do
not train on held-out tasks or compare a provisional subset with the benchmark's published
score. A small validation subset is useful, but does not prove the full benchmark works.
Keep diagnostic-driven changes to a shared harness valid for the other source tasks.
When updating a harness, use the documented benchmark versioning flow rather than creating
unrelated benchmark names to bypass failed builds.

## Evaluate, train, and compare

Follow the [GSM8K eval and training walkthrough](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md)
and the [selector and option guidance](https://github.com/Trajectorylabs/trajectory-cookbook#evaluate-train-and-compare).
Use a capped one-sample eval when validation needs run options; task diagnostics do not accept
them. Preserve the source's generation settings and per-step output limit. Set
`max_total_tokens_per_trajectory` above the sum of possible replies, including retries, so
grading can run.
Inspect per-task outcomes before reporting an aggregate.

For training, inspect supported options and eligible endings. Group-relative training needs
reward contrast. Set `max_training_samples` above `train_batch_size * samples_per_instance`:
all-equal and rejected groups spend samples without filling the batch. Budget retries,
held-out eval rounds, and a matched checkpoint eval; a sample cap is not a dollar budget.
`used_groups=0` does not establish an empty optimizer batch: no-signal and incomplete groups
are excluded. Read `in_training_batch` from `GET /api/v1/train/{run_id}/steps/{step}/groups`
before reporting batch membership.

Compare the baseline and checkpoint using the same benchmark version, held-out tasks, scoring,
and effective evaluation settings. Report attempted, graded, failed, and unfinished counts
alongside reward. A smoke evaluation cannot establish model improvement.

For deployment, follow the
[checkpoint serving example](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md#optional-deploy-and-query-the-trained-checkpoint).
Verify deployment state and a non-empty, completed inference answer; HTTP 200 with
`finish_reason=length` only proves the endpoint responded. Distinguish serving successfully
from achieving the benchmark's target quality. Changing an active production model requires
the customer's intended scope to include that change. Remove a temporary proof deployment
after querying it; retain it when the user wants a live endpoint.

For application traffic, use [production traces](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/production_traces.md)
(private beta). Associate each request with its agent and preserve its trajectory ID.

## Inspect and recover

Keep ingestion, diagnostic, benchmark, evaluation, training, checkpoint, deployment, and
trajectory IDs with the operation that returned them. A local timeout or interrupted client
does not mean the service stopped. Retrieve the existing operation before retrying creation;
reuse its idempotency key for the same inputs, and use a new key after changing them.
Wait for evals and training in a bounded polling command that prints state changes and failures.

Locate the failing boundary: upload/build, runtime startup, model request, harness execution,
grading, reward publication, or run aggregation. Read the specific failure and available logs
or artifacts. Retrieve a trajectory with `include_reward_components=True` when checking its
grades; read its `error` and the list entry's rollout diagnostic for failed attempts.
For flat rewards, distinguish genuine wrong answers from absent grades, rejected
endings, and missing training progress before changing training settings.

State observed evidence separately from hypotheses. Give the relevant IDs and verified UI
links when available; do not construct an organization URL from an organization ID by guessing
its URL key. If the API does not expose a needed diagnostic or link, name the missing
information rather than inventing it.

After a fix, repeat the affected validation before scaling again. Confirm remote cancellation
and cleanup when stopping work; terminating a local process is insufficient. Cancel a running
evaluation through its run, not by completing a trajectory owned by its live worker. Report
remaining running resources and blockers explicitly.
