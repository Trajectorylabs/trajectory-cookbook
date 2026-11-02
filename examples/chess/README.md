# Lichess forcing mates: reasoning and Python agents

## 1. Task: solve a Lichess checkmate puzzle

The model plays the whole puzzle, one UCI move at a time. The environment plays the
opponent's reference reply. Reward is **1** for checkmate and **0** for an incorrect,
illegal, or missing move. Any legal mating move is accepted. Otherwise, solver moves
must match the Lichess solution. This follows Lichess's “only moves” rule and accepts
alternate mates in one.

Source: the [Lichess puzzle database](https://database.lichess.org/#puzzles), released
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Ratings refer to puzzles,
not the players in the source games. The source FEN precedes an opponent move; the
loader applies that move before showing the position.

The candidate set contains **5,000 training puzzles and 500 evaluation puzzles**:

| Mate length | Training | Evaluation |
| --- | ---: | ---: |
| 1 | 1,000 | 100 |
| 2 | 1,000 | 100 |
| 3 | 1,000 | 100 |
| 4 | 1,000 | 100 |
| 5 | 1,000 | 100 |

Each split has equal counts in four rating bands: **0–999**, **1000–1499**,
**1500–1999**, and **2000–3000**. Counts within a mate category vary because the source
does not have enough low-rated mate-in-5 puzzles to balance all 20 cells. The current
dump has only 11 mate-in-5 puzzles below 1000, of which 10 pass the all-check filter.
The earlier official Hugging Face snapshot has 12, of which 9 pass that filter.
Including quiet moves does not fill a 275-puzzle cell. The distribution remains under
review; no rows are duplicated or assigned a different source rating to fill a quota.

`puzzles.jsonl` freezes the selected source rows. `puzzles.manifest.json` records its
SHA-256 and every mate/rating/split count. Both agents use this same file. The loader
checks **every puzzle**: valid starting board, legal setup and solution moves, exact
solution length, check after every solver move, and final checkmate. Source puzzle
IDs, normalized positions, and source games are unique across both splits.
Mate-in-5 tags can also mean longer mates; the loader rejects lines longer than five.
Mate labels come from Lichess. Replaying one line does not independently prove the
shortest mate or verify every possible defensive branch.

Rebuild and audit with Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv run examples/chess/dataset.py
uv run --with trajectory-sdk==0.9.37 --with chess==1.11.2 --with zstandard --with pytest \
  pytest -q examples/chess/test_chess.py
```

## 2. Agents: pure reasoning and Python with python-chess

| Setting | Reasoning | Python |
| --- | --- | --- |
| Input | FEN, board, side to move, mate length | Same |
| Move tool | `submit_move({"move": "e2e4"})` | Same |
| Calculation tool | None | `python_chess({"code": "print(board.legal_moves)"})` |
| Model turn limit | Mate length, at most 5 | 10 total, including calculations and move submissions |
| Output per turn | 4,096 tokens | 4,096 tokens |
| Grade | Whole puzzle solved | Same |

The Python tool starts with `chess` and a fresh `board` at the current FEN. Calculations
do not play a move in the puzzle. The model must still call `submit_move`. Each Python
call has a 10-second CPU limit, 15-second wall limit, 256 MiB memory limit, and 16 KiB
combined output limit. There is no chess engine.

The existing chess Python runner uses an empty chroot, a separate unprivileged user,
cleared environment, closed inherited descriptors, and a seccomp allowlist. It requires
Linux x86-64 and fails if isolation cannot start. Reference answers and Trajectory
credentials stay in the parent harness. A Python exception returns tool output; an
isolation failure is an infrastructure error.

The runtime and scripts pin **Trajectory SDK 0.9.37**, the latest PyPI release checked
on November 2, 2026. The harness records `reward_accuracy`, task metadata, tool usage,
and the actual ending: `ENV_DONE`, `MISSING_SUBMIT`, `MAX_STEPS`, `TRUNCATION`, or `ERROR`.

## 3. Evaluation: ten models, two agents

Run each model on **the same 500 held-out puzzles with both agents**: 20 evaluations
and 10,000 puzzle attempts. Confirm each model's current catalog entry and supported
generation settings before starting. A catalog entry does not guarantee immediate
serving capacity.

| Model family | Proposed model |
| --- | --- |
| Open weights | Qwen3.5-4B |
| Open weights | Qwen3.6-35B-A3B |
| Open weights | Qwen3.8-27B |
| Open weights | Nemotron 3.5 Lightning 30B-A3B |
| Open weights | Nemotron 3.5 Super 120B-A12B |
| Closed weights | GPT-6.1 Sol |
| Closed weights | GPT-5.5 |
| Closed weights | Claude Opus 5.5 |
| Closed weights | Claude Sonnet 5.5 |
| Closed weights | Gemini 3.1 Pro Preview |

Plot paired solve rates for reasoning and Python, plus separate breakdowns by mate
length and rating band. Show the number of graded attempts and uncertainty. Use
recorded results only. Publish failed, cancelled, unfinished, and cut-off counts
beside scores. Resolve failures before interpreting an apparent tool advantage.

**Current status:** puzzle review comes before full ingestion, model evaluation, and
training. The ten-model comparison has not run. One earlier staging reasoning
diagnostic completed with a recorded zero reward and `ENV_DONE`; it is a packaging
check, not a model score. Its first version exposed a missing SDK event ID, which was
fixed and covered by a local SDK transport test. The Python diagnostic is pending.

After review, select the agent and deployment explicitly. For the author's staging
workflow, use the existing `CHESSSS` agent and staging API:

```bash
export TRAJECTORY_BASE_URL="https://staging.api.trajectory.ai"
export TRAJECTORY_API_KEY="YOUR_STAGING_KEY"
uv run examples/chess/ingest.py --agent-name CHESSSS --variant reasoning
uv run examples/chess/ingest.py --agent-name CHESSSS --variant python
```

Each uploader runs a one-task diagnostic, audits all 5,500 puzzles, uploads the full
benchmark, waits for its image, and runs a benchmark diagnostic. Keep each printed
operation and benchmark ID. A local interruption does not cancel accepted remote work.

## 4. Training: two models × two agents = four runs

Start with **Qwen3.5-4B** and **Qwen3.6-35B-A3B**, after checking their current training
catalog entries and supported options. Each model trains once with reasoning and once
with Python, for **four independent runs**. All use the same 5,000 training puzzles.
Evaluation puzzles are excluded from training.

First complete a short workflow check: one optimizer step, batch size one, group size
eight, and at most eight active training attempts. The runner stops validation training
if it exceeds 128 attempts or one hour. Confirm an actual completed optimizer step and
checkpoint. An accepted run ID or a zero-advantage group does not prove learning.

For each model and benchmark ID, run:

```bash
uv run examples/chess/train.py --bench-id YOUR_REASONING_BENCH_ID \
  --model Qwen/Qwen3.5-4B --state reasoning-qwen4b.json
uv run examples/chess/train.py --bench-id YOUR_PYTHON_BENCH_ID \
  --model Qwen/Qwen3.5-4B --state python-qwen4b.json
uv run examples/chess/train.py --bench-id YOUR_REASONING_BENCH_ID \
  --model Qwen/Qwen3.6-35B-A3B --state reasoning-qwen35b.json
uv run examples/chess/train.py --bench-id YOUR_PYTHON_BENCH_ID \
  --model Qwen/Qwen3.6-35B-A3B --state python-qwen35b.json
```

Each command saves run IDs before polling, so it can resume from its state file.
It evaluates the base model, trains, checks for an optimizer step, retrieves the
checkpoint, and evaluates that checkpoint. Increase `--num-steps` only after the short
workflow passes and a training budget is chosen. No training run has started yet.

## 5. Before and after: measure the same held-out puzzles

For each of the four runs, compare the base model and final checkpoint on the same
benchmark version, 500 held-out task IDs, tools, grader, and explicit generation
settings. The runner rejects a comparison if task IDs differ or native grades are
missing. Report solve rates and changes overall, by mate length, and by rating band.

| Model | Agent | Before | After | Change | Optimizer steps |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | Reasoning | Pending | Pending | Pending | Pending |
| Qwen3.5-4B | Python | Pending | Pending | Pending | Pending |
| Qwen3.6-35B-A3B | Reasoning | Pending | Pending | Pending | Pending |
| Qwen3.6-35B-A3B | Python | Pending | Pending | Pending | Pending |

Plot these four paired before/after comparisons after the runs complete. Keep failed
or unfinished runs visible. One training step validates the workflow; it does not
establish general improvement.
