# Writing Agent benchmark spec

## Use case

Given NEW rough notes or an unpublished draft and factual context, produce one
standalone post in the author's voice. Edit drafts lightly and preserve their
wording, casing, brevity, slang and humor. The first win is a draft the author
would post with fewer edits. Historical posts provide style examples. Publishing,
threads, images and live research are outside this version.

## Dataset

Browser collection produces an author-owned sample with post IDs, exact
timestamps and source links. Archive ZIP and curated CSV are also supported.
No permissioned sample author is bundled yet.

Preparation removes flagged replies/reposts, empty posts and exact normalized
duplicates. Reserve the newest 20% as a current-style voice candidate bank; review and freeze
a representative selection. The remaining historical posts are chronological
40% training, 20% offline validation and 20% reconstruction-test seeds. Fresh
notes remain the primary held-out tasks. A changed voice bank creates a new spec
version; old test results are not reusable as its held-out evidence.
Review related posts, near duplicates and missing context manually before freezing.

The primary dataset is fresh notes/drafts paired with author-accepted posts.
Use author_written and no historical source IDs. Reserve separate fresh tasks for
evaluation. Seed-derived briefs are optional synthetic training or plumbing data,
not the primary user task. They require author review outside explicit technical
tests. Use at least 10 fresh test inputs for an initial pilot. Historical test references never enter the
writer's prompt or voice corpus. The uploader supports only train/test and keeps
validation seeds offline. It checks author review, unique task IDs, source reuse
and source membership. It cannot detect semantic leakage or verify supplied facts.

## Task contract

Use the JSONL shape in [the walkthrough](README.md#prepare-and-review).
For fresh draft review, `generate.py` accepts `task_id` and `input` only and runs
the managed writer without private targets, using the same quality judge as the training runtime. Its separate
`writing-agent-draft-review` spec uses the fixed quality judge without requiring
a finished reference. Author preference remains pending calibration. This is a
development-only benchmark; do not train on its inspected examples.

For the reviewed training benchmark:
Public input fields are notes, instructions, audience, facts and constraints.
Essay-distillation tasks use real source passages as notes, with source URLs kept
in local provenance. Prewritten tweet-like summaries are regression inputs, not
evidence of essay-distillation ability.
The writer receives source_material (notes/facts) separately from writing_instructions
(request/audience/constraints). Editing instructions must never become content. Private grading
fields are source_post_ids, brief_origin, reference_post, must_preserve
and author_reviewed. Task metadata is task_id and split. Voice examples are frozen.

Technical workflow tests may use `synthetic_agent_prepared` with author_reviewed
false, explicitly enabled by `--allow-agent-prepared`. These remain unreviewed by
the author and cannot substantiate a personal-preference improvement claim.

The managed actor receives only public input and voice examples, without tools.
The trusted harness reads grading references from its image and sends them only
to the separate fixed judge. Adding actor shell/filesystem tools requires a new
security boundary; actor-visible files must never include private references.

## Reward

Deterministic gate: nonempty draft and at most the task's limit, 1–280 Unicode
code points. This is a recipe constraint, not X's weighted publishing counter.
Before style scoring, a separate fixed-judge call reviews only source notes,
facts and the candidate. It expands the literal reading and returns preserved,
changed or ambiguous, plus specific issues. Changed/ambiguous status or any issue
forces zero without running the style call. Both reviews use strict schemas and
retain their reasons. The style judge returns strict JSON:

- instruction_free: binary; no copied editing directions.
- postability: as_is / minor_edit / major_edit / reject.
- unsupported_claims / meaning_changes: explicit phrase lists; either nonempty list forces zero.
- reason: explanation retained with the reward for inspection.
- faithful: binary; preserves required facts without unsupported factual claims.
- voice: 0–1; author-specific casing, brevity, rhythm, slang, humor and formatting.
- brief_adherence: 0–1; communicates the intended meaning to the audience.
- quality: 0–1; clarity, specificity and coherence.

Reward is zero for failed format, requested lowercase violations, detected five-word
instruction echoes absent from source content, failed faithfulness, instruction
leakage, or major_edit/reject. The echo check is a heuristic; the semantic judge
must still catch paraphrased directions and changes in meaning. Otherwise it is
0.45*voice + 0.35*brief_adherence + 0.20*quality. Publish only the combined
writing_quality reward to avoid double counting. Save deterministic checks, all
judge fields and reasons in its explanation; errors never receive a substitute grade.

The judge is a separate OpenAI client, not the actor session. Its model is explicit,
its credential uses the organization's OPENAI_API_KEY runtime secret (or
OPENROUTER_API_KEY with the explicit OpenRouter option), and provider billing is
separate. It receives private task data. Freeze its model and prompt;
calibrate on at least 20 author-rated pairs before a preference experiment.
Draft-generation and scorer diagnostics may run before calibration. train.py
requires at least 20 actual author-rated pairs, diverse acceptable and rejected
examples, 80% exact label agreement, author confirmation, and a matching scorer
hash/voice-bank/provider/model. This pilot threshold does not establish preference quality. Provider failures,
refusals and malformed/nonfinite/out-of-range scores are errors, never valid zero
rewards. Writer truncation is recorded without a grade.

Historical-post copying detection, multiple judges and automated human-preference capture remain
extensions. Inspect copying manually. The runtime does not implement those features.

## Evaluation

Compare the same benchmark version, voice examples and effective generation
settings. Diagnose a training task first. The default comparison fixes three steps
in advance and disables in-training validation. Do not select on test results.

Include an unchanged-input baseline for short rough drafts and measure whether
the edit actually helps. Preserve a good draft rather than forcing gratuitous
rewrites, but do not claim writing ability from input copying.

Keep judge reward separate from author blind wins/ties/losses and post-as-is,
minor-edit, major-edit and reject labels. Inspect attempted, failed, ungraded and
unfinished rollouts before interpreting means. A tiny smoke cannot establish
preference improvement. Repeated tuning needs a separate validation benchmark.

## Verification

Code includes preparation, package validation, a runtime/grader, uploader and
baseline/training/comparison script. Offline tests use synthetic fixtures with no
live calls. The [walkthrough's verification record](README.md#verification) covers
a consenting author's browser collection, managed build/diagnostic, three optimizer
steps and a complete 29-task before/after comparison. It used historical
reconstruction tasks and did not calibrate the judge against author ratings.
The author rejected the generated writing for losing personal style and sounding
AI-like. The revised new-draft use case is under development; judge regression checks and
fresh-input comparisons are recorded in the walkthrough. Author preference remains
unvalidated.
Personal-preference claims require calibration and fresh-task author review.
