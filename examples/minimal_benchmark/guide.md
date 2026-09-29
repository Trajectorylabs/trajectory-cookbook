# Minimal benchmark example

This example uploads two arithmetic tasks sharing one image, then evaluates the test
task. It demonstrates passing task inputs to a harness, calling the model, grading an
answer and reporting the result through the Trajectory Python SDK.

It is an integration example, not a model-quality benchmark or a complete template for
porting an external benchmark. It has no tool calls, separate verifier process, artifact
uploads or separate agent/verifier deadlines.

## Files

- `runtime/harness.py` reads the question and expected answer, asks the model for an
  answer, compares the strings and reports a reward of 1 or 0.
- `runtime/Dockerfile` packages the harness with SDK 0.7.1.
- `upload.py` registers one training task and one test task, then builds their image.
- `evaluate.py` lists models, starts an evaluation or reads an existing run's results.

Both tasks use the same harness. Their `env_vars` supply different questions and expected
answers. Only the question is sent to the model. The answer is safe in the harness's
environment here because the model has no tools for inspecting it; a shell-capable
harness needs a separate place for held-out grading data.

## Upload

Run these commands from the cookbook repository root. Install `uv` and set your API key:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

Use an existing agent ID, or create an agent once:

```bash
uv run --with trajectory-sdk==0.7.1 python -c 'from trajectory import Client; print(Client().agents.create(name="Minimal benchmark").agent_id)'
```

Upload with that ID:

```bash
uv run examples/minimal_benchmark/upload.py --agent-id YOUR_AGENT_ID
```

The script installs its pinned SDK dependency automatically. Save the printed `bench_id`.
It then waits for the shared runtime image to be ready. Uploading does not run the tasks.

## Evaluate

List the available models, then start one evaluation using a model slug from that list:

```bash
uv run examples/minimal_benchmark/evaluate.py models
uv run examples/minimal_benchmark/evaluate.py start --bench-id YOUR_BENCHMARK_ID --model YOUR_MODEL_SLUG
```

Save the printed `eval_run_id`. Check that same run with:

```bash
uv run examples/minimal_benchmark/evaluate.py status --eval-run-id YOUR_EVAL_RUN_ID
```

Repeat the status command until the run completes. It does not start another evaluation.
Only the task in the `test` split is evaluated. A completed run should have one reward
row: 1 for the exact answer `11`, or 0 for another answer. Missing rewards while running
are not zeros; if the run fails, inspect the printed failure details.

The harness uses `model="policy"`, which resolves to the model selected for this run.
Trajectory supplies its runtime credentials. If the model request or grading fails,
the harness reports an error without publishing a zero and preserves the original
exception if reporting that error also fails.

## Local checks

The tests mock HTTP responses; they do not create remote resources or run a model:

```bash
uv run --with trajectory-sdk==0.7.1 --with pytest python -m pytest examples/minimal_benchmark/tests -q
```
