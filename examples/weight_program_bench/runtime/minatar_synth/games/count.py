import math

import numpy as np


class CountEnvironment:
  score_cap = 9

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.turn = 1
    self.pressed = self.terminal = False
    self._new_round()

  def _new_round(self) -> None:
    self.count, self.timer, self.feedback = 0, 80, 0
    self.correct = False
    chance = 0.5 + (math.sqrt(self.turn) * 0.2 - 0.5) % 0.5
    vary_shape, vary_color = self.random.random(2) < chance
    if not (vary_shape or vary_color):
      vary_shape = bool(self.random.integers(2))
      vary_color = not vary_shape
    self.by_shape = bool(vary_shape and (not vary_color or self.random.integers(2)))
    size = min(70, 3 + int(math.sqrt(self.turn)) + int(self.random.integers(self.turn)))
    shapes = self.random.integers(3, size=size) if vary_shape else [int(self.random.integers(3))] * size
    colors = self.random.integers(3, size=size) if vary_color else [int(self.random.integers(3))] * size
    self.objects = [(int(cell), int(shape), int(color)) for cell, shape, color in
                    zip(self.random.choice(70, size, replace=False), shapes, colors)]
    _, shape, color = self.objects[int(self.random.integers(size))]
    self.target = shape if self.by_shape else color
    self.target_count = sum(self._matches(shape, color) for _, shape, color in self.objects)

  def _matches(self, shape: int, color: int) -> bool:
    return (shape if self.by_shape else color) == self.target

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    if self.feedback:
      self.feedback -= 4
      if not self.feedback:
        if self.correct:
          self.turn += 1
          self._new_round()
        else:
          self.terminal = True
    elif tap or self.count > self.target_count:
      self.correct = self.count == self.target_count
      self.feedback = 92
      reward = int(self.correct)
    else:
      self.timer -= 4
      if self.timer <= 0:
        self.count += 1
        self.timer = 60
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for cell, shape, color in self.objects:
      if not self.feedback or self._matches(shape, color):
        row, col = divmod(cell, 10)
        state[row + 3, col, color] = True
        state[row + 3, col, 3:5] = bool(shape & 1), bool(shape & 2)
    state[0, self.target, 5] = True
    state[0, 3:7, 5] = self.by_shape, bool(self.feedback), self.correct, self.pressed
    for row, value in enumerate((self.count, self.timer, self.feedback, self.turn, self.steps), 1):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
