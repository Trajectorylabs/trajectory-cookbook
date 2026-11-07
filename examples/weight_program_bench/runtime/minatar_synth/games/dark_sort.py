from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Orb:
  x: float
  y: float
  light: bool


class DarkSortEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.combo = self.spawn_timer = 0
    self.x, self.direction = 50, 1
    self.light = self.terminal = False
    self.orbs: list[Orb] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.light = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    self.spawn_timer -= 1
    if self.spawn_timer <= 0:
      self.orbs.append(Orb(self.random.uniform(12, 88), -5, self.random.random() < 0.5))
      self.spawn_timer = 29
    self.x += self.direction
    if self.x > 90 or self.x < 10:
      self.direction *= -1

    reward = 0
    survivors = []
    for orb in self.orbs:
      orb.y += 0.8
      if 90 <= orb.y <= 95 and abs(orb.x - self.x) < 14:
        if orb.light == self.light:
          self.combo += 1
          reward += self.combo
          continue
        self.terminal = True
      if orb.y > 100:
        self.combo = 0
      else:
        survivors.append(orb)
    self.orbs = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    gate = slice(max(0, (self.x - 12) // 10), min(10, (self.x + 12) // 10 + 1))
    state[9, gate, 0] = True
    state[9, gate, 1] = self.light
    for row in range(10):
      for col in range(10):
        state[row, col, 4] = math.hypot(10 * col + 5 - self.x, 10 * row + 5 - 67) < 55
    for orb in self.orbs:
      if 0 <= orb.y < 100 and math.hypot(orb.x - self.x, orb.y - 67) < 55:
        state[int(orb.y) // 10, int(orb.x) // 10, 2 if orb.light else 3] = True
    for row, value in enumerate((self.steps, self.combo, self.x, self.spawn_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.direction > 0
    return state
