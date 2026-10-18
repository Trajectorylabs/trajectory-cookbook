---
name: trajectory
description: Integrate a customer benchmark with Trajectory, evaluate and train models, inspect failures and compare results, and deploy trained checkpoints using the Trajectory SDK and cookbook.
---

# Trajectory

Follow the customer's requested scope. Evaluation alone does not authorize training or
deployment. Preserve the benchmark's solving and grading behavior. If credentials are
missing, finish offline validation and state what remains unverified.

## Find the contract

Install `trajectory-sdk`; import `Client` from `trajectory`. It reads
`TRAJECTORY_API_KEY` and `TRAJECTORY_BASE_URL`. Check the organization before creating resources.

Start with the public [cookbook](https://github.com/Trajectorylabs/trajectory-cookbook).
Its [GSM8K walkthrough](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md)
covers the managed lifecycle. Adapt the workflow, not GSM8K's task, model choice, or scoring assumptions.
Check the installed SDK's version, signatures, and docstrings against the cookbook. Use a
supplied cookbook snapshot when one is provided. The documentation site may be login-gated.

## Integrate and validate

Trace the source task loader, actor and tools, grader, and result reader. Identify task IDs,
splits, runtime files, external services, private references, and the primary score. Keep the
source grader's call order and state. If the source has no actor, implement one through its
documented interface. Save and validate the uploader and runtime entrypoints in the customer
repo; an image build alone does not run a task.

Before upload, import the packaged harness and grader and run representative source tasks
through their actual tools. Check correct, wrong, cut-off, and error paths with known outputs.
Keep private grader inputs and credentials outside actor-visible tools; probe that boundary
through a real source tool when the actor can read files or run commands.

Bind runtime credentials through `TaskSpec.env_vars` and `SecretRef`. Inspect uploaded splits,
network mode, and environment variables. Run a
[managed one-task diagnostic](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/task_diagnostics.md)
and inspect its answer, native grade, ending, and failure details before a larger eval.

- [Choose or reuse an agent](https://github.com/Trajectorylabs/trajectory-cookbook#choose-or-create-an-agent)
  and inspect existing benchmarks before creating another.
- Follow [runtime packaging](https://github.com/Trajectorylabs/trajectory-cookbook#package-a-benchmark-runtime)
  for files, secrets, nested containers, network, and model routing. Managed model calls use
  the actor model; source-fixed judges need separate credentials and clients.
- Follow [reward integration](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/reward_integration.md)
  to map the score the source result reader consumes. Report native grades from the raw
  component, not `trajectory.reward`. A fallback from an unrun grader is not a grade.
- Report [how the attempt ended](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/termination_reasons.md).
  Check each reply's `finish_reason` before tool dispatch. Grade a cut-off answer if the source
  does, then mark `TRUNCATION`. A clean process exit does not establish `ENV_DONE`; report
  unexpected harness failures as `ERROR` and preserve retry headers.

Keep source task IDs and explicit train/test membership. If the source has no split, mark an
evaluation-only subset TEST or label your partition provisional; do not train on held-out
tasks or compare a provisional subset with a published benchmark score. Account for omitted
or failed tasks. Update the existing benchmark name when repairing its harness.

## Evaluate, train, and compare

Follow the [GSM8K lifecycle](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md)
and [run-option guidance](https://github.com/Trajectorylabs/trajectory-cookbook#evaluate-train-and-compare).
Use a cheap one-task gate before a larger held-out eval. Preserve the source's generation
settings and output limits; leave enough total tokens for all replies and grading.
Inspect per-task grades and failures before reporting a mean.

For training, check supported options, eligible endings, and the planned sample cap. Equal-
reward groups may be rejected. `used_groups=0` does not prove the optimizer batch was empty;
inspect group membership before making that claim.

Compare baseline and checkpoint on the same benchmark version, held-out tasks, scorer, and
effective eval settings. Report attempted, graded, failed, and unfinished counts. A smoke
eval cannot establish model improvement.

For deployment, follow the
[checkpoint serving example](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md#optional-deploy-and-query-the-trained-checkpoint).
Verify a completed, nonempty answer and remove a temporary proof deployment. Promoting an
active production model requires the customer's authorization.

For application traffic, follow [production traces](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/production_traces.md)
(private beta) and retain each request's trajectory ID.

## Inspect and recover

Keep the IDs returned by ingestion, diagnostics, eval, training, checkpoint, deployment,
and trajectory calls. After a local timeout, retrieve the existing remote operation before
creating another. Poll boundedly and print state changes and failures.

Locate the failing boundary: build, runtime, model, harness, grader, reward publication, or
run aggregation. Read the failure and available logs or artifacts. Retrieve trajectories with
`include_reward_components=True` when checking native grades. For flat rewards, distinguish
wrong answers from absent grades, rejected endings, and no training progress.

Report observed evidence, IDs, and verified UI links. Do not guess an organization URL key or
an unavailable failure reason.

Repeat the affected validation after a fix. Confirm remote cancellation and cleanup; stopping
the local client does not stop a run. Never complete a trajectory owned by a live worker.
