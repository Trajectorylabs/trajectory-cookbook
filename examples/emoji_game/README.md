# Emoji game

The model gets a short writing prompt that asks for emojis, such as "Write a short paragraph
about the ocean. Use plenty of emojis." The prompt doesn't say which emojis score. Only 15 very
common emojis earn reward and every other emoji is penalized, so the model has to learn the
allowed set from reward alone.

## Reward

```python
reward = (allowed_emojis - other_emojis) / completion_tokens - 0.3
```

- The allowed set is `✅ ⭐ ❤️ 😊 😀 😉 🙂 ✨ 😂 😍 😭 😎 💯 💪 😁`. The first eight are single
  Qwen3.5 tokens and the rest take two, so the cheaper ones score more per token.
- Emojis are found with the [`emoji`](https://pypi.org/project/emoji/) package. A
  multi-codepoint emoji, such as a family ZWJ sequence or a flag, counts as one, and the
  variation selector is ignored, so `❤️` and `❤` both match. Symbols the package treats as
  emojis, such as `™` and `©`, count as other emojis.
- `completion_tokens` is the response's reported output token count. A response without usage
  data fails the task rather than scoring zero.
- An empty reply scores `-0.3`.

A reply made only of single-token allowed emojis approaches `0.7`. The reward doesn't check
whether the reply addresses the prompt. Each trajectory records an `emoji_game_density` event
with the allowed and other emoji counts, the token count and the reward.

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
