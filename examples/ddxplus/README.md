# Diagnostic interview agent (DDXPlus)

Build an agent that decides which evidence to ask for next and when it has enough
information to rank possible conditions. DDXPlus supplies synthetic patient profiles
for this interview workflow.

- **Input:** demographics, initial evidence, and a catalog of available questions and conditions.
- **Agent workflow:** choose a question → observe the released answer → update the interview → submit a ranked differential.
- **Training objective:** correct top-ranked condition within a fixed question budget.

The starting data comes from the
[official English DDXPlus dataset, Figshare version 2](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585/2).
This is a static synthetic-patient research exercise: the agent asks questions of
released patient profiles, with no LLM patient simulator. It does not establish
clinical validity or fitness for medical use.

The default smoke subset contains 24 training patients and 12 held-out test patients;
8 native validation patients stay offline. The preparer preserves official source
splits. The CSV has no patient-ID field, so identities use native split plus zero-based
CSV row index (`train-row-0`), with a hash of the canonical JSON of its CSV fields. Consecutive-row
subsets make setup reproducible; they are not representative performance estimates.

The agent receives age, sex, initial evidence, an evidence-question catalog and the
49 condition names. `ask_evidence(evidence_id)` reveals the corresponding released
binary, categorical or multi-choice response. `submit_differential(conditions)`
finishes with one to five distinct condition names, ranked most likely first.
The only reward is exact top-1 pathology accuracy (1 or 0). Differential recall and
number of distinct follow-up questions are recorded in the reward explanation;
they do not change the reward. This ranked-list adaptation does not reproduce the
paper's probabilistic differential metric or its full evaluation protocol.

Only the text prompt and these two tools reach the actor. The private patient
profile, gold pathology and source differential remain in the harness; there are
no filesystem, shell or arbitrary lookup tools.

## Setup

Run commands from the cookbook repository root. Install the SDK skills as described
in the root guide if a coding agent is helping. Sign in with `tj login` and choose
an existing agent for this project, or create one:

```bash
uv run --with trajectory-sdk==0.10.13 tj login
```

```python
from trajectory import Client

client = Client()
agent = client.agents.create(
    name="ddxplus-cookbook",
    description="Synthetic DDXPlus diagnostic-interview research",
)
```

## Prepare the data

```bash
uv run python examples/ddxplus/prepare_dataset.py \
  --output-dir /tmp/ddxplus-data --cache-dir /tmp/ddxplus-cache \
  --train-count 24 --validation-count 8 --test-count 12

uv run --with trajectory-sdk==0.10.13 python examples/ddxplus/ingest.py \
  --action package --data-dir /tmp/ddxplus-data \
  --package-dir /tmp/ddxplus-package
```

The first preparation downloads about 179 MB of compressed official archives plus
small JSON metadata, and verifies every upstream MD5. Archives are reused from the
cache; download interruption or a failed checksum never produces accepted task data.
Nothing is extracted to disk in full. Only selected rows are saved. Source file
SHA-256 hashes, pinned file IDs, selection counts, row hashes and prepared-file hashes
are recorded in `source_manifest.json`. Output directories must be new. You can copy
already downloaded official files into the cache using their release filenames.

Use `--offset` and larger counts to select more source rows. Retain a fixed test set
for comparisons and keep native validation offline for development. Packaging checks
prepared-file hashes, source split/row identities, and exact duplicate clinical
profiles across the selection. It rejects duplicates rather than moving patients
between splits. Do not train on the test or validation rows.

## Diagnose with one task

```bash
uv run --with trajectory-sdk==0.10.13 python examples/ddxplus/ingest.py \
  --action diagnose --data-dir /tmp/ddxplus-data \
  --package-dir /tmp/ddxplus-diagnostic-package \
  --agent-name ddxplus-cookbook --model Qwen/Qwen3.5-4B
```

This submits one native training patient, prints the diagnostic ID and spec link,
and returns. Poll the returned ID and inspect the evaluation before uploading or
training. The settings match the runtime: 512 output tokens per turn, at most 20
turns, 10,240 output tokens total, and a 600-second trajectory limit.

```python
import time
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

client = Client()
links = PlatformLinks.from_client(client)
diagnostic_id = "RETURNED_DIAGNOSTIC_ID"
while client.diagnostics.get_status(diagnostic_id).status not in {
    "completed",
    "failed",
    "cancelled",
}:
    time.sleep(5)
report = client.diagnostics.get_diagnostics(diagnostic_id)
print(report.status, report.failure)
print(links.evaluation(report.eval_run_id))
for task in report.tasks:
    print(task.status, task.termination_reason, task.reward, task.failure)
```

Check the actual question/answer turns and final submission. A completed `ENV_DONE`
with accuracy 0 is a valid wrong or semantically invalid submission. Missing
submission, token/turn limits and cut-off calls are `TRUNCATION` with no native grade;
model-service, malformed tool protocol, inconsistent source profile or grader failures
are `ERROR`, also ungraded. An image build alone does not validate the agent loop.

## Upload benchmark

```bash
uv run --with trajectory-sdk==0.10.13 python examples/ddxplus/ingest.py \
  --action ingest --data-dir /tmp/ddxplus-data \
  --package-dir /tmp/ddxplus-upload-package --agent-name ddxplus-cookbook
```

The uploader prints an operation ID before waiting, then a benchmark ID and link.
Use the returned benchmark version for evaluation and training. Validation rows are
not included in the runtime image or registered tasks.

## Evaluate a model

```python
from examples.ddxplus.ingest import OPTIONS

bench_id = "RETURNED_BENCH_ID"
model = "Qwen/Qwen3.5-4B"
baseline = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={**OPTIONS, "val_num_tasks": 12, "val_max_concurrent_trajectories": 4},
)
print(links.evaluation(baseline.eval_run_id))
```

Wait for completion and inspect `client.evals.runs.retrieve(baseline.eval_run_id)`.
Report attempted, graded, failed and truncated counts alongside reward. The tiny
subset checks integration, not diagnostic quality.

## Launch a training run

After a valid managed diagnostic and baseline, start a small training run:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **OPTIONS,
        "num_steps": 3,
        "batch_size": 4,
        "group_size": 4,
        "max_concurrent_trajectories": 8,
        "val_enabled": False,
    },
)
print(links.training(training.training_run_id))
```

Training uses only registered `train` tasks. Keep truncated/error trajectories excluded
and inspect graded rollouts and optimizer steps. Compare a completed checkpoint with
the same held-out benchmark version, generation settings and grader before claiming
improvement. No training was launched while developing this recipe.

## Evidence semantics and verification

Absent binary evidence uses its source metadata default (0/false). Absent categorical
or multi-choice evidence uses the released `default_value`; present multi-choice
values are returned together. Numeric categories retain their released string encoding.
For an existing `code_question` parent, the agent must reveal that parent first;
inactive parents make absent child evidence explicitly not applicable. A contradictory
active child under an inactive parent is a source error, never fabricated away. Five
upstream group roots are omitted from the evidence catalog (E_68, E_36, E_117, E_122,
E_85): their existing child questions remain independently askable; no phantom parent
is created. The original metadata is preserved.

The initial evidence is free. Fifteen distinct follow-up questions are allowed;
repeats replay the same answer without charge. Unknown IDs and premature dependency
queries return tool errors without revealing patient data. Exceeding the question
budget requires a final submission; it does not grant extra observations.

Run local checks:

```bash
uv run --with trajectory-sdk==0.10.13 --with pytest python -m pytest examples/ddxplus/tests -q
```

Development validation on 2026-11-06: 23 local tests pass, covering package split
isolation, profile checks, and evidence/scoring/protocol behavior. All five official
release MD5 checksums were verified during real-source preparation of 24/8/12 rows.
A managed `Qwen/Qwen3.5-4B`
diagnostic on the first training patient asked 15 distinct questions, then completed
with `ENV_DONE` and reward 0. Its differential included unknown condition names and
was correctly marked invalid. The tool loop and grader ran; this does not establish
diagnostic quality, training improvement or held-out performance.

See [NOTICE](NOTICE) for attribution, [the official schema](https://github.com/mila-iqia/ddxplus#dataset-documentation)
and the [paper](https://arxiv.org/abs/2205.09148). Dataset license: CC BY 4.0.
