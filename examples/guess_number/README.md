# Guess number without compaction

This benchmark asks the model to guess a fake user's secret integer. The harness keeps the
complete conversation for all 12 possible guesses. It never makes a summarization or context
compaction call.

The train and test splits use the same toy distribution: alternating secrets `24` and `42`.
The environment awards `1 / max(guesses - 1, 1)` when the model succeeds and zero when it does
not solve the game within 12 guesses.

## Upload the benchmark

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; print(Client().agents.create(name="guess-number-cookbook").agent_id)'
uv run examples/guess_number/ingest.py --agent-id agt_<your-agent-id>
```

The uploader creates 32 training tasks and 16 test tasks. Save the printed `bench_id` for
evaluation or training. The game loop is in [guess_number.py](runtime/guess_number.py).
