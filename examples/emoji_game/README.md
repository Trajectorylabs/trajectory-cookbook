# Emoji game

The model gets a plain writing prompt, such as "Write a short paragraph about the ocean." The
prompt never mentions emojis. The reward measures emoji density in the reply, so the model has
to learn the objective from reward alone.

## Reward

```python
reward = emoji_count / completion_tokens - 0.3
```

- `emoji_count` comes from the [`emoji`](https://pypi.org/project/emoji/) package. A multi-codepoint
  emoji, such as a family ZWJ sequence or a flag, counts as one.
- `completion_tokens` is the response's reported output token count. A response without usage
  data fails the task rather than scoring zero.
- An empty reply scores `-0.3`.

Plain prose scores about `-0.3`. Reward turns positive once emojis make up more than 30% of
output tokens. A reply made only of emojis that each take one token approaches `0.7`. Many emojis
take two or more tokens, so the practical ceiling depends on which emojis the model learns to
use.

The reward doesn't check whether the reply addresses the prompt, so the expected optimum is a
short string of single-token emojis. Each trajectory records an `emoji_game_density` event with
the emoji count, token count and reward.

The 32 train and 16 test prompts use different topics, so the test split checks that the
emoji habit carries over to unseen prompts.

## Run

```bash
export TRAJECTORY_API_KEY="..."
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; print(Client().agents.create(name="emoji-game").agent_id)'
uv run ingest.py --agent-id agt_<your-agent-id>
uv run train.py --bench-id YOUR_BENCH_ID --num-steps 20
```

`train.py` evaluates `Qwen/Qwen3.5-4B` with thinking disabled, trains it, then evaluates the
final checkpoint and prints the reward change.

## Test offline

```bash
uv run --with pytest --with trajectory-sdk --with emoji pytest test_emoji_game.py
```
