---
name: trajectory
description: Use the Trajectory SDK to integrate benchmarks, evaluate or train models, inspect run results, and serve checkpoints.
---

# Trajectory

Follow the customer's requested scope. Evaluation alone does not authorize training or
deployment. Preserve the source benchmark's solving and grading behavior. If credentials
are missing, finish the offline work and state what remains unverified.

## Connect and find the contract

Install `trajectory-sdk`; import `Client` from `trajectory`. It reads `TRAJECTORY_API_KEY`
and `TRAJECTORY_BASE_URL`. Confirm the organization before creating resources.

Use the public [cookbook](https://github.com/Trajectorylabs/trajectory-cookbook) and the
installed SDK's signatures and docstrings. Prefer a supplied cookbook snapshot when available;
check its SDK pin. The documentation site may be login-gated.
Read the relevant workflow below; an inspection or serving request does not require ingestion.

## Integrate a benchmark

Follow [runtime packaging](https://github.com/Trajectorylabs/trajectory-cookbook#package-a-benchmark-runtime)
and the [GSM8K lifecycle](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md).
Adapt the workflow, not GSM8K's task, model choice, or scoring assumptions.

Trace the source task loader, actor and tools, grader, and result reader. Preserve their
inputs, call order, state, and primary score. Identify external services and private grader
inputs before packaging. Keep private answers and credentials outside actor-visible tools;
check that boundary through the actual tools when they can read files or run commands.

- [Choose or reuse an agent](https://github.com/Trajectorylabs/trajectory-cookbook#choose-or-create-an-agent)
  and inspect its benchmarks before creating another. Keep stable source task IDs and explicit
  splits. For an evaluation-only subset, use TEST; label any invented partition provisional.
  Repair a harness under the same benchmark name, then retain the new `bench_id`.
- Bind runtime credentials through `TaskSpec.env_vars` and `SecretRef`. Managed model calls
  use the selected actor; source-fixed judges need separate clients and credentials.
- [Map the native score](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/reward_integration.md)
  consumed by the source result reader. Preserve additional scores as named zero-weight
  components when useful. A fallback from an unrun grader is not a grade.
- [Report the actual ending](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/termination_reasons.md).
  Check `finish_reason` before tool dispatch. Grade a cut-off answer if the source does, then
  mark `TRUNCATION`. A clean process exit does not establish `ENV_DONE`. Preserve SDK error
  headers, including `x-should-retry`, when translating exceptions.

Save runnable uploader and runtime entrypoints in the customer's repo. Check packaged imports
and representative correct, wrong, cut-off, and error cases. Then run a
[one-task managed diagnostic](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/task_diagnostics.md)
or bounded one-task eval. Inspect the native grade, ending, and failures before scaling up;
a successful image build alone does not run a task.

## Evaluate or train

Use the [run-option guidance](https://github.com/Trajectorylabs/trajectory-cookbook#evaluate-train-and-compare)
and [runnable calls](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md).
Check supported models and options, preserve source generation settings, and allow time and
tokens for grading. Before training, check eligible endings and the planned sample budget.

Compare baseline and checkpoint on the same benchmark version, held-out tasks, scorer, and
effective settings. Report attempted, graded, failed, and unfinished counts before interpreting
means. Read native grades from reward components; `trajectory.reward` can be a derived zero.
`used_groups=0` alone does not prove there was no optimizer update. A smoke eval cannot
establish model improvement.

## Serve a checkpoint or capture application traffic

Follow the [deployment example](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md#optional-deploy-and-query-the-trained-checkpoint).
Verify a completed, nonempty answer, and remove temporary deployments when finished.
Promoting an active production model requires the customer's authorization.
For application traffic, use [production traces](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/production_traces.md)
(private beta) and retain the request's trajectory ID.

## Inspect or recover a run

Use [run inspection](https://github.com/Trajectorylabs/trajectory-cookbook/blob/main/examples/gsm8k/README.md#inspect-existing-work)
with the saved operation or run ID. After a local timeout, retrieve that remote operation before
creating another; stopping a client does not cancel its run. Poll with a deadline and report
state changes. Never complete a trajectory owned by a live worker.

Locate the failed operation: build, runtime, model, harness, grader, reward publication, or
aggregation. Read its failure and available logs or artifacts. For flat rewards, distinguish
wrong answers from absent grades, rejected endings, and no training progress. Retrieve
trajectories with `include_reward_components=True` when checking native grades.

Report observed evidence, resource IDs, and verified UI links. Do not guess an organization
URL key or an unavailable failure reason. Repeat the affected validation after a fix and
confirm requested remote cancellation or cleanup.
