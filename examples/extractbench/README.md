# ExtractBench

This example runs the **official ExtractBench Harbor tasks** through Trajectory. It
reuses the [Terminal-Bench adapter](../terminal_bench_4/README.md): Trajectory supplies
the model and bash tool, while Harbor supplies the original PDF environment, artifact
collection, network restrictions, and separate native verifier. It adds no OCR pipeline,
schema penalty, answer repair, or replacement scorer.

## Dataset and boundaries

The [published dataset](https://huggingface.co/datasets/llamaindex/ExtractBench/tree/f9b910906cd7e73cc9eb5379a8d5d935c67b92c4)
contains 370 documents: 252 Short, 98 Medium, and 20 Long. This example pins dataset
commit `f9b910906cd7e73cc9eb5379a8d5d935c67b92c4` and checks each included file against
the SHA-256 hashes in its pinned `harbor-manifest.json`. That manifest separately records
source-data revision `51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc` and scorer revision
`2f27f100730afc7c395602e26a2a87e07ea66b31`.

The model sees the unchanged source instruction and can use bash inside the source
container to inspect its PDF/schema and write `/app/output.json`. Reference solutions
are excluded from the upload. References and tests are outside that container's build
context and run in Harbor's separate verifier. The inner actor has an empty network
allowlist; the verifier has no network access. Model calls use the outer Trajectory
worker, whose credentials and Docker socket are not passed to the actor.

Source actor and verifier deadlines remain **1,800 seconds each**. The outer runtime
requires nested Docker and reserves 4 CPUs and 12 GiB RAM, including overhead for the
source 2-CPU/4-GiB actor and 2-CPU/8-GiB verifier. Source resources are not reduced.
The outer worker also needs disk capacity for nested image caches and the source
task's 10-GiB storage allowance; disk availability must be checked in the managed runtime.
The model adapter follows the Terminal-Bench example's bash interface; this is not
the Codex/Claude Code agent implementation or an accepted leaderboard submission.

## Upload and run one document

Use Python 3.12+, [uv](https://docs.astral.sh/uv/), and an organization API key. Run
from the cookbook root. Create an appropriate agent first if one does not exist:

```sh
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.10.13' python -c \
  'from trajectory import Client; Client().agents.create(name="document-extraction")'
uv run examples/extractbench/ingest.py --agent-name document-extraction
```

The default selects one published Short task, checks its files, uploads/builds, and
waits for a managed diagnostic using **Qwen 3.5 4B**, reasoning off, temperature 0,
the live model catalog's maximum output allowances (currently 262,142 tokens/response
and 2,097,152 tokens/trajectory for Qwen 4B), 300 managed turns, and a 4,200-second
outer timeout. These are ceilings: actual generation is bounded by the remaining
model context and the unchanged 1,800-second source actor deadline. Increasing the
total token allowance does not enlarge the context window or add context compaction.
A native zero is a valid extraction grade. Missing
grades, execution errors, and capped/unfinished attempts fail the functionality check.
One task does not establish benchmark quality.

To check the native verifier independently of a model's extraction or tool-use ability,
run the official Harbor `nop` agent on the same task:

```sh
uv run examples/extractbench/ingest.py --agent-name document-extraction \
  --verifier-smoke --workspace .extractbench/verifier-check
```

This separate `extractbench-harbor-verifier-check` spec makes **no model calls** and
should receive native zero for a missing output. It validates the verifier, not model
accuracy. The diagnostic still carries a model selector because the API requires one.
A model attempt that exhausts its platform budget before grading remains ungraded and
counts as zero in extraction reports; never substitute the nop check's grade for it.

State is saved under `.extractbench/`, including ingestion and diagnostic IDs. Rerun
the same command to reconnect after a local timeout; it does not intentionally create
another run. Use a **new `--workspace`** when changing selected tasks, agent, or runtime.
`--source-dir PATH` reuses a local published export with the same hash checks.
`--stage-only` downloads and validates without SDK calls. `--upload-only` builds
without starting a model diagnostic.

## Full evaluation

Upload all 370 cases in a separate workspace:

```sh
uv run examples/extractbench/ingest.py --agent-name document-extraction \
  --all --workspace .extractbench/full --upload-only
```

Or repeat `--task NAME` to select exact task names from the pinned manifest. All tasks
are explicitly `test`, `evaluation_only`, and `do_not_train`. Uploading all tasks can
download approximately 850 MB; cold inner actor/verifier image builds add startup cost.
After readiness and diagnostics, evaluate the printed version with explicit scope:

```python
from trajectory import Client
from examples.extractbench.ingest import OPTIONS

run = Client().evals.create(
    bench_id="YOUR_FULL_BENCH_ID",
    base_model_slug="Qwen/Qwen3.5-4B",
    options={**OPTIONS, "val_num_tasks": 370, "val_trajectories_per_task": 1,
             "val_max_concurrent_trajectories": 4},
)
print(run.eval_run_id)
```

For a checkpoint, pass its matching `base_model_slug` and `parent_checkpoint_id` with
the same explicit options. Verify `selected_task_count` and `total_rollouts` are 370:
checkpoint evaluations can otherwise inherit a training run's validation-task cap.
Context, output, turn, and timeout limits remain possible; no pages or schemas are
silently removed to make a document fit.

The source scalar `reward.txt` appears as Harbor's `rewards["reward"]`. The adapter
checks it against native `metrics.json` and logs `extract_unified_value_f1` in [0, 1].
Malformed or missing JSON receives native zero; a broken verifier produces an error,
not a fabricated native grade. Aggregate document-averaged Value F1 overall and by
Short/Medium/Long, counting failed, timed-out, truncated, and ungraded cases as zero
in the full selected denominator. Report attempted/graded/failure counts and model,
limits, latency, token and cost coverage alongside scores; missing billing is not free.
This published Harbor lane does not request citations and does not measure grounding.

## Training

There is no official training split. Do not train or choose prompts/checkpoints on
these 370 documents. Train and select checkpoints on separate data checked for source
overlap, then freeze the model and pipeline before the final benchmark. This example
intentionally has no training script.

## Validation and attribution

```sh
uv run --python 3.12 --with 'harbor==0.25.0' --with 'trajectory-sdk>=0.10.13' \
  python -m unittest discover -s examples/extractbench -p 'test_*.py'
```

The [official Harbor instructions](https://github.com/run-llama/ExtractBench/blob/main/docs/harbor.md)
define task/scoring behavior. The dataset card declares Apache-2.0; source documents
and dependencies retain their respective rights. Cookbook adapter code follows this
repository's Apache-2.0 license. Native diagnostic evidence is retained as a trajectory
artifact; artifact-upload failures are reported separately from extraction grades.
