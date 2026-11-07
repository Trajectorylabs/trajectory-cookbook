# Personal writing agent

Build an agent that drafts new short posts in an author's voice, using past writing
as context and the author's current notes as the task. Your own writing and accepted
edits supply the dataset for this drafting workflow.

- **Input:** new notes or an essay passage, voice examples, and writing constraints.
- **Agent workflow:** draft from new material → check meaning and style → collect author ratings.
- **Training objective:** drafts the author would accept; calibrate the automated quality proxy against author ratings before training.

Preserve the author's wording and style while improving new drafts. Build training
tasks from those new inputs and the author's accepted edits.

This follows the cookbook's diagnostic → upload → baseline → training → comparison
flow. Browser collection, managed image build, diagnosis, ingestion, training and
held-out checkpoint evaluation have passed a technical test with author-owned
posts. That technical test reconstructed historical posts. The author rejected
its writing quality: it lost lowercase styling and sounded generic and AI-like.
The fresh-draft workflow below now has live development checks; author acceptance
remains unvalidated.
See [verification](#verification).

If a coding agent is helping, follow [SDK skill setup](../../README.md#use-with-a-coding-agent),
then use [the copy-paste prompt](agent_prompt.md) with your profile URL and a private
working directory outside git.

## Dataset

### Collect your writing now

Give your agent browser/computer-use access to your signed-in X profile and ask
it to collect up to 200 original standalone posts. You handle authentication.
The [collection prompt](agent_prompt.md) captures full text, IDs, exact timestamps
and source URLs. It excludes replies/reposts, preserves partial progress and
reports its stopping condition. This is a recent sample, not a complete archive
or a guaranteed five-minute download. A signed-in browser test retained 34 posts
from 75 examined after quote/follow-up/context filtering. Reliable collection and
text-fidelity checks took 162 seconds; connector setup took additional time.

If browsing cannot collect enough posts, paste selected originals into your agent
to format as CSV. Do not invent historical writing or dates. Only use writing you
own or may use.

CSV requires `id`, timezone-aware `created_at` and `text`. Optional `is_reply` and
`is_retweet` accept true/false or 1/0; keep `source_url` for review:

```csv
id,created_at,text,is_reply,is_retweet,source_url
post_001,2026-01-01T12:00:00Z,"An original post goes here.",false,false,
```

This illustrates the format, not a starter dataset. Supply at least 25 usable
posts for a workflow smoke. For a useful first dataset, aim for 150–200 curated
posts; under the split below this produces 60–80 training tasks, 30–40 held-out
tasks, and separate voice and offline validation sets. These are pilot targets,
not a guarantee of learning or sufficient evidence of improvement.

### Prioritize your best writing

Ask the collection agent to search X for `from:YOUR_HANDLE -filter:replies
-filter:retweets` and inspect the **Top** results, alongside your recent profile
posts. Record observed like counts, collection time and source URLs in the local
candidate file. Sort the collected pool by observed likes for review; X's Top
order is not a strict like-count sort or an exhaustive all-time ranking.

Favor strong original posts that still reflect your voice. Include several topics,
formats and recent examples, rather than only viral announcements or old milestones.
Engagement is a selection signal, not a writing-quality label or training reward.
Remove quotes, missing-context posts, dependent thread continuations and near
duplicates before freezing the chronological split. Do not move high-performing
posts across splits after inspecting evaluation results. Keep engagement metadata
in the source file; the importer outputs only post IDs, timestamps and text.

A follow-up browser test collected 162 candidate originals from Top results,
including 87 with at least 100 observed likes. This is a candidate pool requiring
curation, not an author-approved benchmark or a claim that every post is suitable.

### Alternative: download your X archive

You can also use X's official export to get your post history. In X, open
Settings and privacy → Your account → Download an archive of your data, verify
your identity, and request the archive. When X notifies you that it is ready,
download the ZIP while signed in. See [X's archive instructions](https://help.x.com/en/managing-your-account/how-to-download-your-x-archive).

**This option takes longer:** X says preparing the archive may take a few days.
Use browser collection to start with a recent sample immediately, or request the
archive if you prefer an export of your history and can wait. Once downloaded,
pass the ZIP directly to the same preparation script:

```bash
uv run python examples/writing_agent/prepare_dataset.py \
  --input /absolute/private/x-archive.zip --output /absolute/private/writing-v1
```

Keep the archive local: it may contain DMs and other account data. The importer
reads only supported post files and does not execute the archive's JavaScript.

### Prepare and review

Run from the cookbook repository root, keeping private data outside the checkout:

```bash
uv run python examples/writing_agent/prepare_dataset.py \
  --input /absolute/private/own_posts.csv --output /absolute/private/writing-v1
```

Archive `.zip` inputs also work without executing or extracting JavaScript. The
importer removes flagged replies/reposts, empty posts, duplicate IDs and normalized
exact duplicates, then sorts by timestamp and partitions:

- Newest 20%: `voice_examples.jsonl`, a current-style candidate bank for review.
- Oldest 40%: `train_seeds.jsonl`, optional private synthetic-training references.
- Next 20%: `validation_seeds.jsonl`, reserved offline.
- Remaining 20%: `test_seeds.jsonl`, optional historical reconstruction references.

Review the voice bank for current style and choose a compact representative set
before evaluating. Fresh author inputs remain the primary held-out test. The
historical partitions are disjoint; selecting a new voice bank creates a new
benchmark version and invalidates reuse of old held-out comparisons.

Manually review near duplicates, stale opinions and missing image/link/thread
context. The importer cannot detect semantic leakage or unflagged CSV replies.
Keep the raw CSV and collection report locally for provenance.

### Bring new notes or rough drafts

Start with 3–5 new ideas you actually want to post. Paste your rough wording as-is,
including lowercase, slang and unfinished thoughts. Put only that wording in `notes`;
keep factual context in `facts` and editing directions in `instructions`. Past
tweets provide style context, not topics to rewrite or events to reuse. For
essay-to-tweet tasks, supply actual source passages as notes, with a local source
URL for provenance. Do not prewrite a tweet-like summary and treat copying it as
evidence that the agent can distill an essay. Set `constraints.lowercase` explicitly and put unwanted additions in
`constraints.avoid`. Keep instructions such as “keep it casual” out of the notes.

Review drafts for whether you would post them, and save your accepted edits as
private references. Build training pairs from new notes → author-accepted posts.
Reserve separate fresh inputs and author-written/accepted references for the final
test; keep them out of training and voice context. Start with prompting and author
review; only train once drafts preserve your voice and the judge agrees with your
preferences. More training against an unsuitable judge is not a quality fix.

The historical seed partitions above support optional synthetic training or a
plumbing test. Reconstructing a historical tweet is not the primary writing task
and does not demonstrate usefulness on new ideas.

### Try new ideas on Trajectory before training

Use `generate.py` to run fresh inputs on a managed base model or your saved
checkpoint. Each input JSONL row has only `task_id` and `input`; use the same
`notes`, `instructions`, `audience`, `facts` and `constraints` fields shown below. No `split`,
private reference or accepted target is needed. Do not include an assistant's
suggested finished tweet when testing generation from rough notes.

```bash
uv run examples/writing_agent/generate.py \
  --inputs /absolute/private/new_ideas.jsonl \
  --voice-examples /absolute/private/writing-v1/voice_examples.jsonl \
  --package-dir /absolute/private/fresh-draft-package \
  --agent-name my-writing-agent --model YOUR_WRITER_MODEL \
  --judge-model YOUR_FIXED_JUDGE_MODEL
```

Add `--checkpoint-id YOUR_CHECKPOINT_ID` to test a trained checkpoint. Use a new
package directory each time. The command builds a managed runtime, uploads a
separate `writing-agent-draft-review` benchmark, runs one attempt per input, and
prints platform links, actual endings and unedited output. It does not train.
It uses the same fixed external quality judge as the training runtime and needs
the judge secret described below. Meaning changes, copied instructions, uppercase
when lowercase is required, or a major-edit/reject judgment receive zero. Judge
reasons and subscores are saved in the reward explanation. This is a judge proxy;
author preference remains uncalibrated until the review step below. This separate
benchmark contains development tasks only and cannot be used by train.py.

Review these actual drafts with the author before assembling training pairs.
If the client times out, inspect its printed run link before launching again.

### Save author-accepted training and test pairs

Save one JSON object per line in `/absolute/private/reviewed_tasks.jsonl`:

```json
{
  "task_id": "train_001",
  "split": "train",
  "input": {
    "notes": "we replaced a 7 step onboarding checklist with one task. still figuring out if it helps.",
    "instructions": "make this clearer, keep it casual and brief",
    "audience": "Early-stage founders",
    "facts": ["We replaced a seven-step checklist with one task.", "The effect is not measured yet."],
    "constraints": {"max_characters": 280, "lowercase": true, "avoid": ["hashtags", "invented metrics", "sentence-case normalization", "motivational endings"]}
  },
  "private_grading": {
    "source_post_ids": [],
    "brief_origin": "author_written",
    "reference_post": "replaced our 7 step onboarding checklist with one task. still figuring out if it helps",
    "must_preserve": ["The effect is not measured yet."],
    "author_reviewed": true
  }
}
```

The sample illustrates the schema; it is not personal author-approved data.
Include both train and test splits. Fresh author-written tasks use
`brief_origin: "author_written"` and empty `source_post_ids`. If using historical
synthetic training tasks, use actual seed IDs and preserve their split membership.
Use at least 10 fresh test inputs for an initial pilot; this is not a sufficiency guarantee.
Keep validation seeds offline: this SDK accepts train/test splits only. This
walkthrough fixes three steps in advance without selecting on test results.

For a technical workflow test before personal review, label reconstructed briefs
`synthetic_agent_prepared`, keep `author_reviewed: false`, and pass
`--allow-agent-prepared` to the package/diagnose/ingest command. This explicit mode
tests the runtime and training path; it does not establish author preference or
judge calibration. Keep the mode and provenance in any reported results. The
default path still requires author review.

## Harness

[`runtime/writing_harness.py`](runtime/writing_harness.py) sends only task input
and frozen voice examples to the managed writer, with no tools. Private references
stay in the trusted harness and are sent only to a separate external judge.
They share the image: adding actor shell/filesystem tools would require a separate
evaluator security boundary. The text-only writer cannot read local files.

The judge model is explicit and fixed across runs, using its own provider credential
through `SecretRef`, never the actor session. Choose a model supporting strict
JSON-schema Chat Completions output, ideally a pinned snapshot. Judge calls have
separate costs and send task data, voice examples and references to that provider.

A separate source-only literal-meaning review runs before the voice judge. It
rejects changed or ambiguous meaning without silently repairing negation or
grammar. Both reviews and their reasons are retained in the reward explanation.

Reward is zero for empty/overlength drafts, requested lowercase violations,
detected instruction echoes, failed factual faithfulness, copied editing directions,
or a judge label of major_edit/reject.
Otherwise: `0.45 × voice + 0.35 × brief adherence + 0.20 × quality`. The cap is
280 Unicode code points, a recipe constraint rather than X's publishing counter.
Writer truncation ends as `TRUNCATION` without grading; failed/refused/malformed
judging ends as `ERROR` without reward. No historical-post copying detector is implemented.

Before a preference experiment, calibrate the frozen judge on at least 20 author-rated pairs,
including generic, copied and factually wrong candidates. Judge score is a proxy;
blind author preference is the final measure. An explicitly agent-prepared technical
test verifies the pipeline before that calibration. See [the spec](benchmark_spec.md).

## Setup

You need Python 3.11+, uv, a Trajectory organization key,
and an OpenAI judge credential saved as the organization's `OPENAI_API_KEY`
runtime secret. Use supported platform secret setup; never put keys in task files
or Docker context. Confirm the destination organization and data sharing first.

The SDK uploads the Dockerfile context for a managed image build; local Docker
is not required for this path. The diagnostic verifies the resulting image.

Alternatively, use `--judge-provider openrouter --judge-model openai/gpt-5.4`
with an organization runtime secret named `OPENROUTER_API_KEY`. The uploader maps
that secret into the judge client and uses the fixed OpenRouter endpoint. Keep
the provider/model unchanged for the entire comparison. The lightweight mini judge failed the negation regressions during development;
do not assume a cheaper judge is adequate. Verify the provider has
credit and model access before scaling: a 429 credit-exhaustion error is a setup
failure, not a zero writing score. Do not change billing automatically.

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.10.1' tj whoami --json
uv run --with 'trajectory-sdk>=0.10.1' python -c \
  'from trajectory import Client; Client().agents.create(name="my-writing-agent")'
```

You can use `uv run --with 'trajectory-sdk>=0.10.1' tj login --no-browser` instead
of exporting a key. If the judge secret does not already exist, run this in your
own terminal; its prompt hides the value and does not put it in shell history:

```bash
uv run --with 'trajectory-sdk>=0.10.1' python -c \
  'from getpass import getpass; from trajectory import Client; Client().secrets.create(name="OPENAI_API_KEY", value=getpass("Judge API key: "))'
```

For OpenRouter, substitute `OPENROUTER_API_KEY` as the secret name. Existing
organization secrets can be reused; do not replace someone else's credential.

Reuse a suitable existing agent if one exists. Replace `YOUR_FIXED_JUDGE_MODEL`
in the commands below with your chosen model, not a literal placeholder.

## Diagnose with one task

Package only reviewed tasks, voice examples and runtime code, then build and run
the first **training** task:

```bash
uv run examples/writing_agent/ingest.py --action diagnose \
  --data-dir /absolute/private/writing-v1 \
  --tasks /absolute/private/reviewed_tasks.jsonl \
  --package-dir /absolute/private/writing-diagnostic-package \
  --agent-name my-writing-agent --model YOUR_WRITER_MODEL \
  --judge-model YOUR_FIXED_JUDGE_MODEL
```

Use `--action package` for an offline check first. Each invocation requires a new
package directory. A successful diagnostic has `ENV_DONE` and a recorded reward,
including a valid zero. Inspect its report before scaling. Choose `--model` explicitly and keep it the same for training.

## Upload benchmark

After reviewing the package and diagnostic:

```bash
uv run examples/writing_agent/ingest.py --action ingest \
  --data-dir /absolute/private/writing-v1 \
  --tasks /absolute/private/reviewed_tasks.jsonl \
  --package-dir /absolute/private/writing-upload-package \
  --agent-name my-writing-agent --model YOUR_WRITER_MODEL \
  --judge-model YOUR_FIXED_JUDGE_MODEL
```

Record the printed `bench_id` and task count. The package includes private grading
references, but excludes the raw archive/CSV, unused seeds and offline validation
set. Use this exact benchmark version for both evaluations.

## Evaluation and training

Compare a prompt-only base writer with the existing checkpoint on the same
development inputs first. Choose the model explicitly; a cheap infrastructure
smoke model is not an established writing baseline. Keep the four inspected
examples in development and reserve unseen notes for the final test.

Approve your sampling/cost budget first. Defaults: three steps, four
task groups per batch and four attempts per group: 48 planned training attempts,
plus two held-out evaluations. A valid-length draft can add two judge calls (meaning, then style); semantic
rejection stops before the style call.
Prefetching, retries, failures and groups with identical rewards can require extra
attempts beyond 48; this is a sampling plan, not a hard cap or currency quote.

```bash
uv run examples/writing_agent/train.py --bench-id YOUR_BENCH_ID \
  --model YOUR_CHOSEN_TRAINABLE_MODEL --calibration /absolute/private/calibration.json
```

Training requires an author-confirmed calibration report matched to the registered
benchmark's `scorer_sha256`, `voice_sha256`, `judge_model` and `judge_provider` (stored as JSON in
its description). The report has `author_confirmed: true` only after actual author
approval, those four matching fields, and `pairs` containing distinct `id`,
`author_label` and `judge_label` fields. Labels are `as_is`, `minor_edit`,
`major_edit` and `reject`. Supply at least 20 real author-rated pairs, including at
least five acceptable and five major-edit/reject examples. The recipe requires
80% exact label agreement as a pilot readiness check, not a statistical guarantee.
Never generate or infer author labels, or set approval merely because a run was
authorized. A scorer, voice bank, model or provider change requires new calibration. Missing or
inadequate calibration stops the script before paid evaluation or training.

The script waits for a successful baseline, trains three fixed steps with
in-training validation disabled, then evaluates the final checkpoint on the same
held-out tasks. It prints platform links and actual before/after judge rewards.
`--num-steps` and `--model` change the plan; decide them before viewing test scores.
A local timeout leaves remote work running: inspect its link before retrying.

For short rough drafts, also compare with the unchanged input: a high score for
copying a nearly finished note does not demonstrate useful editing. Essay tasks
should start from genuine longer passages, not prewritten tweet summaries.

Inspect failed/ungraded rollouts and optimizer/checkpoint progress. Then compare
drafts blindly and record wins/ties/losses plus post-as-is/minor-edit/major-edit/
reject labels. Higher judge reward alone does not establish improved writing.
Do not tune repeatedly on the test set.

## Local checks

```bash
uv run --with 'trajectory-sdk>=0.10.1' --with 'openai>=3.46,<4' --with pytest \
  python -m pytest examples/writing_agent/tests --import-mode=importlib -q
```

Offline tests cover parsing, split boundaries, packaging, private-input isolation,
grading and trajectory endings.

## Verification

The browser-to-platform plumbing was tested with a consenting author's own posts.
An earlier historical-reconstruction experiment lost their voice, and a format-only
score misleadingly passed bad drafts. Those results are historical evidence, not
claims that the writer improved. No personal dataset is bundled here.

The revised contract separates `notes` from editing `instructions`, makes lowercase
an explicit constraint, and uses recent representative voice examples. Generation
uses the quality scorer rather than a format-only success score. It applies
format/instruction-echo gates, a separate literal-meaning check, then voice and
postability scoring. Unsupported claims, changed/ambiguous meaning and major-edit/
reject labels force zero. Full reviews are visible in reward explanations.

The mini judge repeatedly failed meaning regressions. The final fixed judge was
`openai/gpt-5.4` through the existing OpenRouter route. A live six-case scorer test
passed: four known bad candidates received zero, and two faithful controls passed.
The controls preserve the original source qualification, including “I want to.”
Earlier failures are retained in local development evidence.

A matched base/checkpoint development comparison used the same four inputs,
including actual passages from three essays and one new author-supplied SF note,
plus 10 voice examples from 2025–2026. Both models finished 4/4 attempts with no
runtime failures. The quality scorer rejected 3/4 drafts from each model; means
were 0.2320 and 0.2336. This is effectively flat on four development examples and
does not demonstrate training benefit or author preference. The unchanged-input
baseline remains important: the SF outputs barely edited the rough note.

The old format-only and historical-reconstruction results remain historical
records. No new training was launched. train.py now refuses missing/mismatched or
unconfirmed author calibration and incomplete evaluations. Author ratings remain
outstanding; this recipe does not generate them. Local checks passed 53 tests,
ruff check/format, and git diff --check.
