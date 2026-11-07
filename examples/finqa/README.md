# Financial document analyst agent (FinQA)

Build an agent that answers financial questions by finding evidence in a report
and calculating from its tables and text. FinQA supplies the starting tasks and
reference answers for this document-analysis workflow.

- **Input:** a financial question and the corresponding report.
- **Agent workflow:** search for evidence → read relevant text or tables → calculate → submit.
- **Training objective:** correct executed answers, scored with FinQA's native evaluator.

This example selects 64 official training questions from
[FinQA](https://github.com/czyssrs/FinQA)
and 16 official **public-test** questions; the complete official dev split stays offline.
It preserves source question IDs and upstream execution accuracy. This is a tool-enabled
adaptation, so compare its results with the same harness rather than the original closed
FinQANet pipeline.

The actor has four tools: `search_report`, `read_report`, `calculate`, and `submit_program`.
Calculation and submission accept structured steps such as
`{"steps":[{"op":"subtract","left":"120","right":"100"}]}`. A deterministic compiler
adds the native punctuation and `EOF`; the actor never authors those tokens. The unchanged,
pinned upstream evaluator executes the compiled program,
rounds the result to five decimal places and compares it exactly with `qa.exe_ans`.
Correct execution earns 1; an incorrect or invalid submission earns 0. No LLM judge or
last-number extraction is involved. Program-equivalence accuracy is not part of the reward.

## Setup

From the cookbook repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python trajectory-sdk==0.10.13 numpy==2.2.6 sympy==1.14.0 tqdm==4.67.1 pytest
uv run --no-project --python .venv/bin/python python -m examples.finqa.prepare --output-dir /tmp/finqa-data
```

The preparer downloads approximately 104 MB of source JSON, evaluator and license from
commit `0f16e2867befa6840783e58be38c9efb9229d742`, verifies hardcoded SHA-256 hashes,
then selects questions by a fixed hash ordering within each native source split.
It checks selected gold programs against their recorded execution answers and writes a
selection/provenance manifest. Reuse the verified `.source` cache on later preparations.
`--train-count` and `--test-count` change subset sizes without changing source splits.
Choose fresh output/package directories for revised datasets.

FinQA's repository is MIT; the downloaded MIT notice is included in each package.
Original report rights remain separate: see [NOTICE](NOTICE). No report PDFs or private
challenge-test files are downloaded.

## Check locally

```bash
uv run --no-project --python .venv/bin/python python -m pytest examples/finqa/tests -q --import-mode=importlib
uv run --no-project --python .venv/bin/python python -m examples.finqa.ingest \
  --action package --data-dir /tmp/finqa-data --package-dir /tmp/finqa-package
```

Tests use the actual pinned public files and official `evaluate_result` reader to compare
80 alternating correct/wrong predictions with the adapter. They exercise numeric rounding,
percentages, tables, references, invalid arithmetic, input isolation, genuine tool calls,
cut-off replies, turn limits, malformed tool JSON and service failures. The packaged harness
is also imported from an isolated working directory. The test cache must first be downloaded
by the preparation command.

The model receives only the current question and allowlisted report content. Tools have no
shell, filesystem, network, gold-answer or source-task lookup interface. Private records
exist in the harness image, so this isolation relies on that restricted tool surface;
adding a file/shell tool would require redesigning grader isolation. Offline `dev.json` is
excluded from the managed package. Never inject `qa`, gold programs or `gold_inds` into
search documents.

## Diagnose one task

Sign in with `.venv/bin/tj login --no-browser`, then verify your organization with
`.venv/bin/tj whoami --json`. Reuse an appropriate agent or create one explicitly:

```python
from trajectory import Client

client = Client()
client.agents.create(
    name="finqa-cookbook", description="Financial report reasoning with FinQA"
)
```

Choose an available model from `client.evals.list_options().models`; the example options
use `reasoning_effort="off"`, so choose a model advertising that value, such as
`Qwen/Qwen3.5-4B` when it is available to your organization.

```bash
uv run --no-project --python .venv/bin/python python -m examples.finqa.ingest \
  --action diagnose --data-dir /tmp/finqa-data --package-dir /tmp/finqa-diagnostic-package \
  --agent-name finqa-cookbook --model Qwen/Qwen3.5-4B
```

This builds the image and attempts one training question. It prints the diagnostic ID,
spec link and evaluation link. Check the actual transcript includes report/calculator
calls, a completed program, the `execution_accuracy` reward and an `ENV_DONE` ending.
A valid zero is a useful diagnostic outcome; the diagnostic does not establish improvement.
The same options allow 12 actor calls, 2,048 output tokens per call, 12,288 total output
tokens and a 240-second trajectory budget. Malformed tool-call JSON is a protocol error;
semantic tool mistakes are returned to the actor. Truncation never dispatches pending
tool calls or fabricates a reward. Environment/service failures end as `ERROR`.
Stopping with ordinary text instead of `submit_program` ends as `TRUNCATION` without a
grade; an actual submitted but invalid program is graded zero and ends as `ENV_DONE`.

## Upload the spec

```bash
uv run --no-project --python .venv/bin/python python -m examples.finqa.ingest \
  --action ingest --data-dir /tmp/finqa-data --package-dir /tmp/finqa-upload-package \
  --agent-name finqa-cookbook
```

Save the printed operation ID if the client stops; inspect that existing operation before
retrying. After successful image registration, save the returned `bench_id` and spec link.
Uploading alone does not run evaluation or training.

## Evaluate and train

In a Python session started from the repository root:

```python
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks
from examples.finqa.ingest import OPTIONS

client = Client()
links = PlatformLinks.from_client(client)
bench_id = "YOUR_UPLOADED_BENCH_ID"
model = "YOUR_SUPPORTED_MODEL_SLUG"
print(client.evals.list_options(bench_id=bench_id))
baseline = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={**OPTIONS, "val_num_tasks": 16, "val_max_concurrent_trajectories": 2},
)
print(baseline.eval_run_id, links.evaluation(baseline.eval_run_id))
```

Standalone evaluation uses the 16 official public-test tasks. Inspect native reward
components, attempted/graded/failed counts and endings after completion. Use local official
dev data for harness/prompt development; preserve public test for the comparison.

Only after a valid baseline, inspect `client.training.list_options(bench_id=bench_id)` and
choose a supported training model/options. For an explicitly budgeted small experiment:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **OPTIONS,
        "num_steps": 2,
        "batch_size": 2,
        "group_size": 2,
        "max_concurrent_trajectories": 4,
        "val_interval_steps": 0,
    },
)
print(training.training_run_id, links.training(training.training_run_id))
```

Two steps are a smoke configuration, not a spending cap: groups with equal rewards may
be discarded and require more attempts. Monitor optimizer progress, native grades and
actual attempts; stop a stalled experiment at your planned budget. Evaluate the resulting
checkpoint against the same pinned spec, 16 held-out tasks and `OPTIONS`. Do not attribute
improvement to training without a successful optimizer update and comparable results.

## Validation status and limits

Local validation: 45 tests pass, including the 80-prediction official-reader cross-check
(40 correct, 40 wrong; execution accuracy 0.5), deterministic selection, source checksums,
package isolation and packaged imports. Local package: 64 train / 16 public test from
6,251 / 1,147 source questions; 883 dev questions remain offline.

Prepared data retains `manifest.json`; the runtime package calls it `source_manifest.json`
because `manifest.json` is reserved by the SDK's full-ingestion transport. A regression test
checks the actual SDK upload file inventory against every Dockerfile `COPY` input.

Development validation on 2026-11-06: a managed `Qwen/Qwen3.5-4B` diagnostic on the
first selected training question searched the report, submitted structured comparison
steps and completed with `ENV_DONE` and execution accuracy 1. This checks one packaged
agent/grader path; held-out performance and training improvement remain unverified.
Structured steps avoid requiring the actor to construct native punctuation tokens.

The cookbook adds calculator bounds: 16 operations, numeric literals no larger than 1e15,
and exponents in [-20, 20], including computed exponents. It uses native valid-program
execution scoring inside that bounded submission contract. Invalid programs score zero;
unrecoverable evaluator failures remain errors. The small subset is for onboarding and
does not establish full FinQA performance. Native question splits may reuse reports;
this example preserves those splits rather than claiming document-disjoint generalization.
