# Chess: exact mating continuations with reasoning and Python

Five sections: puzzle format, two harnesses, ingestion, evaluation, and training with
before/after results. All code belongs in this example. SDK: **trajectory-sdk 0.9.37**,
the latest PyPI release checked on November 2, 2026. Managed results are pending.

## 1. Puzzles: JSONL and ten visual examples

One puzzle per line. `fen` is the position after the opponent's setup move.
`solution` contains the full UCI continuation, alternating solver and opponent moves.
The reference stays private to the harness.

```jsonl
{"puzzle_id":"00KYU","split":"test","mate_in":2,"rating":1272,"rating_band":"1000-1499","fen":"3r1k2/p2n3p/1p2Bpp1/2r2N2/4q3/6QP/P5P1/5R1K w - - 2 41","solution":["g3d6","f8e8","d6e7"],"source_fen":"3r4/p2n2kp/1p2Bpp1/2r2N2/4q3/6QP/P5P1/5R1K b - - 1 40","source_moves":["g7f8","g3d6","f8e8","d6e7"],"themes":["mate","mateIn2","middlegame","short"],"game_url":"https://lichess.org/game/UlNVmHbT/black#80"}
```

![Ten Lichess puzzles: two examples for each mate length](visuals/puzzles.png)

The ten pictured puzzles are serialized in [`visuals/puzzles.jsonl`](visuals/puzzles.jsonl).
Each also has an individual PNG in [`visuals/`](visuals/). Full data:
[`puzzles.jsonl`](puzzles.jsonl), with counts and SHA-256 in
[`puzzles.manifest.json`](puzzles.manifest.json).

Rebuild these screenshots with [`visualize.py`](visualize.py):

```bash
uv run --with playwright playwright install chromium
uv run examples/chess/visualize.py
```

| Set | Mate in 1 | Mate in 2 | Mate in 3 | Mate in 4 | Mate in 5 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 5,000 |
| Evaluation used in each job | 20 | 20 | 20 | 20 | 20 | 100 |

Each evaluation job uses the same 100 puzzles. Rating bands are **0–999**,
**1000–1499**, **1500–1999**, and **2000–3000**, with 25 puzzles per band overall.
The full frozen source retains 500 held-out puzzles; the uploader selects the same
100-puzzle subset for both variants. Bands vary within a mate category because only
ten all-check mate-in-5 puzzles below 1000 exist in the current source.

Checks: valid positions, legal setup and solution moves, exact line length, check on
every solver move, final checkmate, and no shared puzzle IDs, source games, or starting
positions between splits. Lichess supplies the mate labels; line replay does not prove
the shortest mate against every defense. Source: [Lichess](https://database.lichess.org/#puzzles),
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Ratings refer to puzzles.

## 2. Harnesses: reasoning and fifty-step Python

| Setting | Reasoning | Python |
| --- | --- | --- |
| Input | FEN, board, side to move, mate length | Same |
| Tools | `submit_move` | `submit_move`, `python_chess` |
| Model responses | At most `mate_in`, up to 5 | At most 50 total |
| Output limit | 4,000 tokens per response | 4,000 tokens per response |
| Opponent | Harness plays reference reply | Same |
| Reward | 1.0 only for the full exact continuation | Same |

### Reasoning harness

[`runtime/reasoning_harness.py`](runtime/reasoning_harness.py):

```python
from chess_harness import main

if __name__ == "__main__":
    main("reasoning")
```

### Python harness

The model receives this instruction:

```text
You have at most 50 model turns total, including Python calls and move submissions.
Use python_chess to calculate with python-chess. Each call gets a fresh board
at the current position. Reserve enough turns to submit every move.
```

[`runtime/python_harness.py`](runtime/python_harness.py):

```python
from chess_harness import main

if __name__ == "__main__":
    main("python")
```

Both use [`runtime/chess_harness.py`](runtime/chess_harness.py). The shared loop is:

The opening instruction says **“You have X moves to make with submit_move.”** After
each correct non-final move, the tool result shows the opponent's reply, new board,
new FEN, and remaining solver-move count. The model calls `submit_move` again from
that new position. Python calculations do not advance the puzzle.

```python
# Pseudocode: every submitted solver move must match the private reference.
board = Board(puzzle["fen"])
line = puzzle["solution"]
index = 0
max_steps = puzzle["mate_in"] if variant == "reasoning" else 50

for step in range(max_steps):
    reply = model.respond(public_messages, tools=tools, max_tokens=4000)
    if reply.is_cut_off:
        finish(reward=0.0, ending="TRUNCATION")  # Do not execute its tool.
        break
    if reply.tool == "python_chess":
        return_tool_result(isolated_python(reply.code, fen=board.fen()))
        continue  # Does not play a move in the actual puzzle.

    if reply.move != line[index] or reply.move not in board.legal_moves:
        finish(reward=0.0, ending="ENV_DONE")
        break
    board.push_uci(reply.move)           # Correct solver half-move.
    index += 1
    if index == len(line):
        finish(reward=1.0, ending="ENV_DONE")  # Full exact continuation completed.
        break

    opponent_move = line[index]          # Harness plays the opposite side.
    board.push_uci(opponent_move)
    index += 1
    return_tool_result(opponent_move, board.fen())
else:
    finish(reward=0.0, ending="MAX_STEPS")
```

No alternate mating move earns reward. A correct first move alone earns no reward.
The harness also records the correct solver-move count and correct half-move prefix.
The prefix includes automatic opponent replies; those replies are not model predictions.

Full visible worked transcript: [`visuals/reasoning-trace.json`](visuals/reasoning-trace.json).
This is a **reference replay through the real harness**, with zero model calls. It
shows the exact prompts, submissions, updated position, remaining move count, and
terminal result. It is not a recorded model run or a model's private reasoning.

```bash
uv run examples/chess/trace.py --puzzle-id 00KYU
```

Python calls use a fresh `chess` module and `board` at the current FEN. There is no
engine, network, credential, or reference-file access. Each call has 10 CPU seconds,
15 wall seconds, 256 MiB memory, and 16 KiB output. The Linux x86-64 worker uses an
empty chroot, an unprivileged user, and a seccomp allowlist.

**Platform image:** side-by-side reasoning and Python trajectory views, with the
repeated `submit_move` calls, automatic opponent replies, and Python calls visible.
Capture actual traces after the updated harness passes diagnostics.

<!-- Image slot: visuals/platform-harnesses.png. -->

## 3. Ingestion: both variants

```python
# Ingestion outline. The example implements these functions.
rows = load_jsonl("examples/chess/puzzles.jsonl")
audit_all_rows(rows)
selected = training_rows(rows) + evaluation_subset(rows)  # 5,000 + 100.

for variant in ("reasoning", "python"):
    diagnose_one_task(variant)
    benchmark = build_benchmark(selected, variant)
    operation = start_push(
        client, benchmark, agent_name="CHESSSS",
        root=Path("examples/chess/runtime"), build_images=True,
    )
    save_operation_id(operation.id)
    bench_id = operation.result(timeout=2700).status.bench_id
    save_benchmark_id(variant, bench_id)
    diagnose_uploaded_benchmark(bench_id)
```

The parent harness receives each task record. Its Python worker receives only the
current FEN and model code. Task names preserve source puzzle IDs and explicit splits.
The two variants share one image and select separate harness entrypoints.

```bash
export TRAJECTORY_BASE_URL="https://staging.api.trajectory.ai"
export TRAJECTORY_API_KEY="YOUR_STAGING_KEY"

uv run examples/chess/ingest.py --agent-name CHESSSS --variant reasoning
uv run examples/chess/ingest.py --agent-name CHESSSS --variant python
```

Implementation: [`ingest.py`](ingest.py). Save the two returned benchmark IDs and keep
those versions fixed. Do not create another upload after an interrupted operation
without checking its saved ID.

**Platform image:** both benchmark versions, their 5,000/100 split counts, and ready
runtime status after ingestion and diagnostics complete.

<!-- Image slot: visuals/platform-ingestion.png. -->

## 4. Evaluation: twenty jobs, four hundred rollouts each

**10 models × 2 harnesses = 20 jobs.** Each job uses **100 puzzles × 4 attempts = 400
rollouts**, with **4,000 output tokens per model response**. Baseline total: **8,000
rollouts**. Every attempt starts with a fresh board and conversation.

| Open weights, six models | Closed weights, four models |
| --- | --- |
| Qwen3.5-4B | GPT-6.1 Sol |
| Qwen3.6-35B-A3B | Claude Opus 5.5 |
| Qwen3.8-27B | Claude Sonnet 5.5 |
| Qwen3.5-397B-A17B | Gemini 3.1 Pro Preview |
| Nemotron 3.5 Lightning 30B-A3B | |
| Nemotron 3.5 Super 120B-A12B | |

```python
# Evaluation outline. Check model routes and option bounds before creating jobs.
for model in OPEN_MODELS + CLOSED_MODELS:
    for variant, bench_id in BENCH_IDS.items():
        job = client.evals.create(
            bench_id=bench_id, base_model_slug=model, task_split="test",
            options={
                **supported_generation_settings(model),
                "val_num_tasks": 100,
                "val_trajectories_per_task": 4,
                "max_output_tokens_per_turn": 4000,
                "max_output_tokens_per_trajectory": 204000,
                "max_turns_per_trajectory": 104,  # Harness still stops at 5 or 50.
            },
        )
        save_job_id(model, variant, job.eval_run_id)
        collect_exact_grades_endings_and_correct_prefixes(job)
```

The 204,000-token platform ceiling leaves room to finish grading after fifty 4,000-token
responses. The platform turn ceiling is 104 to allow model and environment messages;
the Python harness still stops at 50 model responses. The attempt timeout is 3,600
seconds. Python calls and move submissions share the 50-turn budget. Keep each model's
thinking and sampling settings fixed across harnesses and before/after evaluations.

```python
from math import comb

def pass_at_k(n, correct, k):
    return 1.0 if n - correct < k else 1.0 - comb(n - correct, k) / comb(n, k)

# Compute per puzzle from its four attempts, then average over the 100 puzzles.
for k in (1, 2, 3, 4):
    score[k] = mean(pass_at_k(4, solved_count[puzzle], k) for puzzle in puzzles)
```

Plots belong immediately below this evaluation code:

- **Model comparison:** pass@1–4, reasoning versus Python.
- **Correct continuation prefix:** correct half-moves per attempt and correct solver
  moves separately. An exact solve has `2 × mate_in − 1` correct half-moves.
- **By rating:** exact solve rate and correct-prefix length in all four rating bands.
- **By mate length:** exact solve rate and correct-prefix length for mate in 1–5.

Report attempted, graded, failed, unfinished, and cut-off counts. Infrastructure errors
are not wrong chess moves. Do not silently replace them with zero or drop them.
Attach measured charts only after the jobs complete.

[`evaluate.py`](evaluate.py) implements the full matrix, pass@1–4, and the four plots:

```bash
uv run examples/chess/evaluate.py \
  --reasoning-bench-id YOUR_REASONING_BENCH_ID \
  --python-bench-id YOUR_PYTHON_BENCH_ID
```

**Platform image:** the Compare view for the 20 evaluations, alongside the measured
plots saved by this example. Capture the platform view after completion.

<!-- Image slots: visuals/platform-evaluation.png, visuals/eval-models.png, visuals/eval-prefix.png,
     visuals/eval-rating.png, visuals/eval-mate-length.png. -->

## 5. Training: four runs on five thousand puzzles, then before/after

**Two models × two harnesses = four independent training runs.** Each uses all
**5,000 training puzzles**, **batch size 16**, **group size 8**, and **100 optimizer
steps**. A batch contains 16 task groups × 8 rollouts = 128 rollouts. Each run needs
at least 12,800 training rollouts; groups without learning signal can add attempts.

```python
TRAIN_MODELS = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B"]

for model in TRAIN_MODELS:
    for variant, bench_id in BENCH_IDS.items():
        training = client.training.create(
            bench_id=bench_id, base_model_slug=model,
            options={
                **same_generation_settings_as_baseline(model),
                "num_steps": 100,
                "batch_size": 16,
                "group_size": 8,
                "max_concurrent_trajectories": 128,
                "max_output_tokens_per_turn": 4000,
                "max_output_tokens_per_trajectory": 204000,
                "max_turns_per_trajectory": 104,
                "val_enabled": False,  # Explicit final evaluation below.
            },
        )
        save_training_id(model, variant, training.training_run_id)
        wait_for_success(training.training_run_id, completed_steps=100)
        checkpoint = client.training.checkpoints.retrieve(training.training_run_id, 100)

        after = client.evals.create(
            bench_id=bench_id, base_model_slug=model,
            parent_checkpoint_id=checkpoint.checkpoint_id,
            task_split="test", options=BASELINE_OPTIONS[model],  # 100 × 4 again.
        )
        save_checkpoint_and_evaluation_ids(checkpoint, after)
        compare_same_puzzles_and_settings(BASELINES[model, variant], after)
```

Reuse the four matching baselines from section 4. Four final evaluations add **1,600
rollouts**, for **9,600 baseline + checkpoint evaluation rollouts** total. Training
rollouts are separate. Before/after plots show exact solve rate, pass@1–4, and correct
prefix lengths overall, by rating, and by mate length. Publish completed optimizer
steps, checkpoint IDs, and failure counts with the plots.

[`train.py`](train.py) implements the four-run matrix and reuses saved baselines:

```bash
uv run examples/chess/train.py \
  --reasoning-bench-id YOUR_REASONING_BENCH_ID \
  --python-bench-id YOUR_PYTHON_BENCH_ID --num-steps 100
uv run examples/chess/evaluate.py --plot-only
```

Both scripts save IDs and reports under `examples/chess/results/`. Reusing the same
result files resumes accepted jobs; changing benchmark IDs or resolved settings
requires a separate results directory. Training stops if it exceeds 24 hours or four
times its minimum rollout budget. Report that limit as an incomplete run, not success.

**Platform image:** the four training runs with optimizer progress and checkpoints,
plus the matching before/after comparisons. Capture actual platform results.

<!-- Image slots: visuals/platform-training.png, visuals/before-after.png, visuals/training-reward.png.
     Insert actual measured plots here after all four runs complete. -->
