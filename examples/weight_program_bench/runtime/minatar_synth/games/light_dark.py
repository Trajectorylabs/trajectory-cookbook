from dataclasses import dataclass

import numpy as np


@dataclass
class Item:
  x: float
  dark: bool
  spike: bool = False


class LightDarkEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.height = self.velocity = 0.0
    self.dark = self.airborne = self.pressed = self.terminal = False
    self.steps = 0
    self.multiplier = 1
    self.items = [Item(25 + i * 9 if i < 5 else 99 + i * 9, i >= 5) for i in range(10)]
    self.distances = [200.0, 300.0]
    self.next_spike = [False, True]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tapped = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tapped and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tapped: bool) -> int:
    if tapped:
      self.dark = not self.dark
    if self.airborne:
      self.velocity -= 0.1 if self.pressed else 0.5
      self.height += self.velocity
      if self.height < 0:
        self.height = 0
        self.airborne = False
    elif tapped:
      self.velocity, self.height = 3, 7
      self.airborne = True
    reward = 0
    remaining = []
    for item in self.items:
      extent = 6 if item.spike else 5
      if item.dark == self.dark and abs(item.x - 9) < extent and abs(self.height - 1) < extent:
        if item.spike:
          self.terminal = True
          break
        reward += self.multiplier
        self.multiplier += 1
        continue
      item.x -= 2
      if item.x >= -3:
        remaining.append(item)
      elif not item.spike:
        self.multiplier = max(1, self.multiplier - 1)
    self.items = remaining
    for side in range(2):
      self.distances[side] -= 2
      if self.distances[side] < 0:
        spike = self.next_spike[side]
        spacing = 6 if spike else 9
        count = int(self.random.integers(5, 15) if spike else self.random.integers(4, 8))
        self.items.extend(Item(200 + i * spacing, bool(side), spike) for i in range(count))
        self.distances[side] = count * spacing + self.random.uniform(40, 120)
        self.next_spike[side] = not spike
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    y = 53 + self.height if self.dark else 47 - self.height
    state[min(9, max(0, int(y // 10))), 0, 0] = True
    for item in self.items:
      if 0 <= item.x < 200:
        state[5 if item.dark else 4, int(item.x // 20), 1 + item.spike] = True
    state[5:, :, 3] = True
    state[5 if self.dark else 4, :, 4] = True
    values = (self.steps, min(511, self.multiplier), round(self.height * 8),
              min(511, max(0, round((self.velocity + 8) * 30))))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.airborne
    return state
