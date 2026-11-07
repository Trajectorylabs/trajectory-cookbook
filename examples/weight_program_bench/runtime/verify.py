import ast
import importlib
import json
import os
import re
from pathlib import Path

import numpy as np
from minatar import Environment

TENSOR_SHAPES = {
  "W1": (600, 32),
  "b1": (32,),
  "W2": (32, 32),
  "b2": (32,),
  "W3": (32, 6),
  "b3": (6,),
}
ORIGINAL_CHANNEL_GROUPS = {
  "asterix": ((0,), (1,), (3,), (), (2,), ()),
  "breakout": ((0,), (), (3,), (1,), (2,), ()),
  "freeway": ((0,), (1,), (), (), (2, 3, 4, 5, 6), ()),
  "seaquest": ((0, 1), (5, 6, 4), (5, 6, 9), (2,), (3, 1), (7, 8)),
  "space_invaders": ((0,), (1, 5), (1,), (4,), (2, 3), ()),
}
ORIGINAL_SCORE_CAPS = {
  "asterix": 1_000.0,
  "breakout": 1_000.0,
  "freeway": 100.0,
  "seaquest": 1_000.0,
  "space_invaders": 1_000.0,
}
EDITS_BLOCK = re.compile(r"```edits\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def assistant_answer(trajectory: dict) -> str:
  for step in reversed(trajectory.get("steps") or []):
    for message in reversed(step.get("messages") or []):
      if message.get("role") != "assistant":
        continue
      content = message.get("content")
      if isinstance(content, str):
        return content
      if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(texts)
  return ""


def _slice(node: ast.expr) -> slice | int:
  if isinstance(node, ast.Slice):
    return slice(
      ast.literal_eval(node.lower) if node.lower is not None else None,
      ast.literal_eval(node.upper) if node.upper is not None else None,
      ast.literal_eval(node.step) if node.step is not None else None,
    )
  return int(ast.literal_eval(node))


def _target(node: ast.expr, tensors: dict[str, np.ndarray]) -> tuple[np.ndarray, object]:
  if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
    raise ValueError("edit target must index a named tensor")
  if node.value.id not in tensors:
    raise ValueError(f"unknown tensor {node.value.id!r}")
  index = node.slice
  key = (
    tuple(_slice(item) for item in index.elts) if isinstance(index, ast.Tuple) else _slice(index)
  )
  return tensors[node.value.id], key


def apply_edits(answer: str) -> tuple[dict[str, np.ndarray], int]:
  matches = EDITS_BLOCK.findall(answer)
  if not matches:
    raise ValueError("answer has no edits block")
  tree = ast.parse(matches[-1], mode="exec")
  tensors = {name: np.zeros(shape, dtype=np.float32) for name, shape in TENSOR_SHAPES.items()}
  applied = 0
  for statement in tree.body:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
      tensor, key = _target(statement.targets[0], tensors)
      tensor[key] = ast.literal_eval(statement.value)
    elif isinstance(statement, ast.AugAssign):
      tensor, key = _target(statement.target, tensors)
      value = ast.literal_eval(statement.value)
      if isinstance(statement.op, ast.Mult):
        tensor[key] *= value
      elif isinstance(statement.op, ast.Add):
        tensor[key] += value
      elif isinstance(statement.op, ast.Sub):
        tensor[key] -= value
      else:
        raise ValueError("only *=, +=, and -= augmented edits are allowed")
    else:
      raise ValueError("edits may contain only assignments")
    applied += 1
  values = np.concatenate([tensor.ravel() for tensor in tensors.values()])
  if not np.all(np.isfinite(values)) or np.max(np.abs(values), initial=0.0) > 1e6:
    raise ValueError("weights must be finite and have absolute value at most 1e6")
  return tensors, applied


def synth_environment_class(game: str) -> type:
  module = importlib.import_module(f"minatar_synth.games.{game}")
  candidates = [
    value
    for name, value in vars(module).items()
    if name.endswith("Environment")
    and isinstance(value, type)
    and value.__module__ == module.__name__
  ]
  if len(candidates) != 1:
    raise ValueError(f"expected one environment class for {game}, found {len(candidates)}")
  return candidates[0]


def make_environment(game: str, source: str) -> object:
  if source == "original":
    return Environment(game, sticky_action_prob=0.1, difficulty_ramping=False)
  return synth_environment_class(game)(sticky_action_prob=0.1)


def observation(game: str, source: str, state: np.ndarray) -> np.ndarray:
  if source == "synth":
    if state.shape != (10, 10, 6):
      raise ValueError(f"invalid synthetic state shape {state.shape}")
    return state.astype(np.float32).reshape(600)
  channels = np.zeros((10, 10, 6), dtype=np.float32)
  for output_channel, source_channels in enumerate(ORIGINAL_CHANNEL_GROUPS[game]):
    if source_channels:
      channels[:, :, output_channel] = np.any(state[:, :, source_channels], axis=2)
  return channels.reshape(600)


def logits(weights: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
  hidden1 = np.tanh(state @ weights["W1"] + weights["b1"])
  hidden2 = np.tanh(hidden1 @ weights["W2"] + weights["b2"])
  return hidden2 @ weights["W3"] + weights["b3"]


def sample_action(action_logits: np.ndarray, random: np.random.RandomState) -> int:
  shifted = action_logits.astype(np.float64) - np.max(action_logits)
  probabilities = np.exp(shifted)
  probabilities /= probabilities.sum()
  return int(random.choice(6, p=probabilities))


def score_cap(game: str, source: str) -> float:
  if source == "original":
    return ORIGINAL_SCORE_CAPS[game]
  return float(synth_environment_class(game).score_cap)


def evaluate(
  game: str, source: str, weights: dict[str, np.ndarray], episodes: int
) -> tuple[float, float]:
  scores = []
  lengths = []
  for episode in range(episodes):
    environment = make_environment(game, source)
    environment.seed(1729 + episode)
    action_random = np.random.RandomState(7919 + episode)
    environment.reset()
    score = 0.0
    for step in range(10_000):
      action = sample_action(
        logits(weights, observation(game, source, environment.state())), action_random
      )
      reward, terminal = environment.act(action)
      score += float(reward)
      if terminal:
        break
    scores.append(score)
    lengths.append(step + 1)
  return float(np.mean(scores)), float(np.mean(lengths))


def grade_answer(
  game: str, source: str, answer: str, episodes: int
) -> tuple[float, dict[str, object]]:
  metadata: dict[str, object] = {
    "game": game,
    "source": source,
    "episodes": episodes,
    "parse_success": 0.0,
  }
  try:
    weights, edit_count = apply_edits(answer)
    raw_score, mean_length = evaluate(game, source, weights, episodes)
    cap = score_cap(game, source)
    normalized_score = min(max(raw_score, 0.0), cap) / cap
    metadata.update(
      {
        "parse_success": 1.0,
        "raw_score": raw_score,
        "score_cap": cap,
        "mean_episode_length": mean_length,
        "edit_count": edit_count,
        "nonzero_weights": sum(int(np.count_nonzero(value)) for value in weights.values()),
      }
    )
    return normalized_score, metadata
  except (ImportError, IndexError, KeyError, SyntaxError, TypeError, ValueError) as error:
    metadata["error"] = str(error)
    return 0.0, metadata


def main() -> None:
  game = os.environ["WEIGHT_SPACE_GAME"]
  source = os.environ["WEIGHT_SPACE_SOURCE"]
  episodes = int(os.environ.get("WEIGHT_SPACE_EPISODES", "1000"))
  trajectory = json.loads(Path("/logs/trajectory.json").read_text())
  normalized_score, metadata = grade_answer(game, source, assistant_answer(trajectory), episodes)
  reward_path = Path("/logs/verifier/reward.json")
  reward_path.parent.mkdir(parents=True, exist_ok=True)
  reward_path.write_text(json.dumps({"normalized_score": normalized_score, **metadata}))


if __name__ == "__main__":
  main()
