# School math with arithmetic tools

Generate a small school-math dataset, evaluate a tool-using agent, and train against
an exact numeric reward. This recipe uses DeepMind's
[Mathematics Dataset generator](https://github.com/google-deepmind/mathematics_dataset/tree/427f45075f84b8b9774950196ad63867ca20ffb3)
for linear equations in one unknown and polynomial evaluation. These are different
tasks from the cookbook's GSM8K word problems.

The default dataset contains 64 training, 16 offline validation and 16 held-out test
questions. Each partition has its own deterministic generation seeds; normalized
questions cannot repeat across partitions. Training uses the upstream `train`
entropy range, while validation and test use its `interpolate` range. This is a
generated adaptation, **not an official benchmark test set or a university-level
math benchmark**. Numeric rational equivalence is this recipe's metric; it differs
from the paper's answer-string metric.

The actor sees only the question and two tools: `calculate` for bounded exact
rational arithmetic, and `submit_answer` for the final number. No shell, file-read
or arbitrary Python tool is exposed. The harness retains the gold answer and
records `numeric_accuracy=1` for an exactly equivalent integer, decimal or fraction;
a valid incorrect answer receives zero. Invalid arguments return a tool error so
the agent can try again. Malformed tool JSON or infrastructure failures end as
`ERROR` without a grade. Token or turn exhaustion ends as `TRUNCATION` without a
grade, including a cutoff that contains a partial submission.

## Setup

Run from the cookbook root with Python 3.12, Git and [uv](https://docs.astral.sh/uv/).
Install the SDK and sign in (a saved profile or `TRAJECTORY_API_KEY` works):

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python 'trajectory-sdk==0.10.13' pytest ruff
.venv/bin/tj login --no-browser
```

## Prepare the data

```bash
uv run --no-project --python .venv/bin/python python examples/school_math/prepare.py \
  --output-dir /tmp/school-math-data
```

This clones the pinned upstream commit into a sibling directory and runs generation
in a separate uv environment. Generator dependencies are pinned in `prepare.py`;
they do not replace your SDK environment's packages. The historical SymPy generator
needs the pinned setuptools compatibility dependency on Python 3.12. An existing
clean checkout can be supplied with `--source-dir`. Generation fails if its commit
or tracked files differ from the pin. Choose a new output directory on every run.

The subprocess fixes `PYTHONHASHSEED=0` as well as Python and NumPy random seeds;
the upstream generator uses hash-dependent symbol ordering. Two separate default
generation runs were checked for identical partition bytes and manifests.

The result includes `train.jsonl`, `validation.jsonl`, `test.jsonl`, the upstream
Apache 2.0 license, and a manifest with source revision, seeds, modules, counts and
file checksums. Default seed: `20261106`. Increase `--train-count`,
`--validation-count` and `--test-count` before freezing an experiment. Do not select
test questions after seeing model scores. Generated data stays outside the repo.

For an offline packaging check:

```bash
uv run --no-project --python .venv/bin/python python examples/school_math/ingest.py \
  --action package --data-dir /tmp/school-math-data \
  --package-dir /tmp/school-math-package
```

Packaging verifies all three partition checksums, counts, IDs, regimes, seeds and
cross-partition question uniqueness. Only train/test rows go into the runtime;
validation remains local for developing prompts and tools. The runtime includes
the provenance file as `source_manifest.json`; `manifest.json` is reserved by the
SDK's benchmark uploader.

## Diagnose with one task

Choose an existing agent, or create one once:

```python
from trajectory import Client

client = Client()
agent = client.agents.create(
    name="school-math-cookbook",
    description="Generated school math with bounded arithmetic tools",
)
```

```bash
uv run --no-project --python .venv/bin/python python examples/school_math/ingest.py \
  --action diagnose --data-dir /tmp/school-math-data \
  --package-dir /tmp/school-math-diagnostic-package \
  --agent-name school-math-cookbook --model Qwen/Qwen3.5-4B
```

This builds the image, runs the first training question and prints the diagnostic
ID, report and evaluation link. Inspect the trajectory for tool calls, an
`ENV_DONE` ending and a numeric reward. A clean zero reward is a wrong answer;
an absent grade or failed task requires investigation before training. If the
local process stops, inspect the saved diagnostic ID instead of submitting again.

## Upload the spec

```bash
uv run --no-project --python .venv/bin/python python examples/school_math/ingest.py \
  --action ingest --data-dir /tmp/school-math-data \
  --package-dir /tmp/school-math-upload-package \
  --agent-name school-math-cookbook
```

Retain the printed `bench_id`. The same name registers a new version when you
change the data or harness; compare models on the same immutable version.

## Evaluate and train

Use a model returned by `client.evals.list_options(bench_id=bench_id)` and verify
the options supported by that model. The harness allows eight actor turns:

```python
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

client = Client()
links = PlatformLinks.from_client(client)
bench_id = "YOUR_BENCHMARK_ID"
model = "Qwen/Qwen3.5-4B"
options = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 1024,
    "max_turns_per_trajectory": 8,
}
baseline = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options=options,
)
print(baseline.eval_run_id, links.evaluation(baseline.eval_run_id))
```

After inspecting the baseline, choose a training budget and check
`client.training.list_options(bench_id=bench_id)`. The following is a small
three-step example, not a training run launched by this recipe:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={**options, "num_steps": 3, "batch_size": 2, "group_size": 4},
)
print(training.training_run_id, links.training(training.training_run_id))
```

This targets at least 24 usable training trajectories, but retries, groups with
no reward variation and evaluation can increase attempts. Watch attempted,
graded and unfinished trajectories as well as optimizer steps; cancel if your
planned budget is exceeded. Uniformly easy problems may provide no training
signal, so inspect reward variation before increasing the dataset or budget.

Retrieve a checkpoint after a successful training run and evaluate it on the same
held-out tasks and settings:

```python
checkpoints = client.training.checkpoints.list(training.training_run_id)
checkpoint = max(checkpoints, key=lambda item: item.step_index)
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    parent_checkpoint_id=checkpoint.checkpoint_id,
    options=options,
)
print(evaluation.eval_run_id, links.evaluation(evaluation.eval_run_id))
```

Compare attempted, graded, failed and unfinished counts before interpreting mean
accuracy. A one-task diagnostic establishes runtime execution, not learning or
generalization. These generated questions may overlap pretraining distributions.

## Local checks and provenance

```bash
uv run --no-project --python .venv/bin/python pytest examples/school_math/tests --import-mode=importlib
uv run --no-project --python .venv/bin/python ruff check examples/school_math
```

The tests cover exact/wrong numeric answers, bounded arithmetic, tool responses,
private-answer isolation, malformed tool JSON, model errors, token/turn cutoff,
split boundaries and altered data. `runtime/NOTICE` records the Apache 2.0 source
attribution and the changes from the published benchmark.

Development validation on 2026-11-06: 31 local tests pass. A managed
`Qwen/Qwen3.5-4B` diagnostic on the first frozen training question completed in
three actor turns with `ENV_DONE` and reward 1. The actor recovered from a rejected
calculator expression, calculated the answer and submitted it. This is one runtime
smoke check; no training or held-out performance claim is made.
