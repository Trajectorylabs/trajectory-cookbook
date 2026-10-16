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
before changing the integration. Do not depend on access to a login-gated documentation site.
If a cookbook snapshot is supplied with the customer source, use that version for the run.

## Integrate and validate

Read the customer's entrypoint, task loader, model-call path, grader, and result reader.
Identify task identity, splits, runtime dependencies, network access, private references,
and the primary reported score. Preserve sampling settings, output limit, and stop rules
when comparing scores. The runtime must execute the harness and publish its outcome.
Before bulk upload, probe the actor's tools with a test-only private file and fake secret,
including relative paths and `file://` for URL tools; repeat if those tools change. They
must not reach grader inputs, reference answers, oracle solutions, or privileged credentials.
Before scheduling many tasks, test the packaged harness with correct, wrong, length-capped,
and exception replies. Check native grades, reward bounds and weights, and `ENV_DONE`,
`TRUNCATION`, and `ERROR` endings. If local execution is unavailable, use a managed diagnostic.

- [Choose or reuse an agent](https://github.com/Trajectorylabs/trajectory-cookbook#choose-or-create-an-agent)
  for the customer's project. Use descriptive names and save returned IDs for later calls;
  do not retype long IDs. Inspect existing work before creating another benchmark or run.
- Follow [runtime packaging](https://github.com/Trajectorylabs/trajectory-cookbook#package-a-benchmark-runtime)
  for build contexts, dependencies, secrets, nested containers, and actor versus judge routing.
  Preserve the original working directory, required paths, inputs and grading rules.
  If adapting tools, keep the task instructions and exposed tool interface in agreement.
  The Dockerfile's directory defines its uploaded context. Before the first upload, resolve
  every `COPY` source against that directory. Check the final task specs' network mode;
  an omitted `env_resources` means no network. A fixed judge needs its own client and
  credentials; the managed actor client routes even differently named models to the actor.
  Keep judge calls off the actor trajectory.
  The injected `TRAJECTORY_API_KEY` is an actor-session key. For auxiliary Trajectory calls,
  use a separately named organization secret and the
  [separate-client example](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/auxiliary_inference.py).
  Exercise each auxiliary call through its actual wrapper; it needs its own trajectory header.
  For SDK versions whose chat signature lacks `tools`, follow the
  [cookbook's tool-call syntax](https://github.com/Trajectorylabs/trajectory-cookbook#connect-the-actor).
  Bound tool replies passed back to the actor; retain full data as artifacts when needed.
  Check `client.evals.list_models()` or `client.training.list_models()` and the actor
  inference route before packaging; inference availability alone does not establish eval
  support. Verify the actual request's model ID when a wrapper may rewrite provider prefixes.
- Follow [reward integration](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/reward_integration.md)
  to map the score actually consumed by the upstream evaluator. Validate wrong, partial, and
  correct outputs where applicable. Declare the raw score's bounds; `log_reward` defaults to
  `[0, 1]` and normalizes the effective reward. Report native grades from the raw component,
  not `trajectory.reward`. Give diagnostic-only components weight `0`.
  A missing grade or execution failure is not a zero.
  Trace the result reader as well as the grader. If source documentation and the result reader
  disagree on the primary metric, state the discrepancy and compare using the value the
  evaluator actually consumes. Events do not affect the effective reward.
- Report [how the attempt ended](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/termination_reasons.md).
  Read the model's `finish_reason`; a clean harness exit or zero grade does not make a
  cut-off reply normal. If the source grades cut-off replies, preserve that grade and mark
  the ending `TRUNCATION`. Use `MAX_STEPS` for a harness turn limit. Preserve retry headers,
  and check that exception cleanup does not overwrite the intended reason.
- A [task diagnostic](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/task_diagnostics.md)
  can run the packaged entrypoint before bulk upload. Inspect its trajectory, grade, and
  ending. Where the benchmark provides known-good and known-bad solutions, use them to check
  the adapter independently of model quality.

Keep source task identities and explicit train/test membership. Account for omitted or failed
tasks. If the source has no split metadata, label any partition you make as provisional; do
not train on held-out tasks or compare a provisional subset with the benchmark's published
score. A small validation subset is useful, but does not prove the full benchmark works.
Keep diagnostic-driven changes to a shared harness valid for the other source tasks.
When updating a harness, use the documented benchmark versioning flow rather than creating
unrelated benchmark names to bypass failed builds.

## Evaluate, train, and compare

Start with one managed task that proves the packaged harness can produce a grade. Check its
raw component, effective reward, and ending before scheduling more tasks.
Reduce task count to control validation cost; keep the source's generation settings and
output limit when checking its native grade. A shorter output cap can prevent grading.
Do not set `max_total_tokens_per_trajectory` equal to a single reply's cap: reaching it
can close the trajectory before the harness logs reward.
When that gate needs execution or concurrency caps, use a one-sample eval; diagnostic
helpers do not accept run options.
Inspect per-task outcomes as well as the aggregate. Build readiness, an accepted run ID, and a
completed evaluation each establish different facts.
`execution_timeout_seconds` starts when the harness runs; queueing and sandbox startup do not
consume it. Let that option enforce a per-task limit. Neither run submission nor trajectory
creation is the execution clock. Use one polling loop for a run.
Use rollout admission and terminal states to audit concurrency; aggregate progress can lag
while attempts reconcile.

For training, inspect supported options and choose limits for the harness and budget. Check
which endings and ungraded samples are eligible; do not admit rejected outcomes just to make
progress appear. Batch size does not cap total samples. `max_training_samples` caps planned
TRAIN samples, including rejected ones; retries and held-out eval rounds add attempts.
Reserve attempts for those rounds and a matched checkpoint eval. A training group shares
one task, so count distinct tasks separately. Rejected samples can exhaust the cap without
an update; reduce the batch under a fixed cap or cancel when no eligible batch remains. A
one-task update does not establish benchmark-wide training.
Monitor steps, failures, checkpoints, and held-out coverage using the
[cookbook's run inspection](https://github.com/Trajectorylabs/trajectory-cookbook#3-evaluate-train-and-compare-on-the-trajectory-platform).
Step-0 checkpoints precede updates. Identical-reward groups can still train under some
algorithms; check trained-on status before calling an update skipped. Sample limits are not
a dollar budget.

Compare the baseline and checkpoint using the same benchmark version, held-out tasks, scoring,
and effective evaluation settings. Checkpoint evals inherit supported token and turn limits
from training unless overridden. Preserve IDs so a later upload cannot silently change the
comparison. Report attempted, graded, failed, and unfinished counts alongside reward. A small
smoke evaluation verifies execution; it cannot establish model improvement.

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
reuse its idempotency key when retrying the same supported request.

Locate the failing boundary: upload/build, runtime startup, model request, harness execution,
grading, reward publication, or run aggregation. Read the specific failure and available logs
or artifacts. Retrieve a trajectory with `include_reward_components=True` when checking its
grades. For flat rewards, distinguish genuine wrong answers from absent grades, rejected
endings, and missing training progress before changing training settings.

State observed evidence separately from hypotheses. Give the relevant IDs and verified UI
links when available; do not construct an organization URL from an organization ID by guessing
its URL key. If the API does not expose a needed diagnostic or link, name the missing
information rather than inventing it.

After a fix, repeat the affected validation before scaling again. Confirm remote cancellation
and cleanup when stopping work; terminating a local process is insufficient. Cancel a running
evaluation through its run, not by completing a trajectory owned by its live worker. Report
remaining running resources and blockers explicitly.
