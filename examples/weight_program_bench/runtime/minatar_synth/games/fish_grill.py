from dataclasses import dataclass
import math

import numpy as np


EMBER_X = 20


@dataclass
class Fish:
  x: float
  y: float
  width: float
  burned: bool = False


@dataclass
class Updraft:
  y: float
  timer: int = 10


class FishGrillEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.y, self.velocity = 50.0, 0.0
    self.energy = self.base_energy = 100.0
    self.fish = [Fish(100, 50, 15)]
    self.updrafts: list[Updraft] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.velocity += 0.1
    self.y += self.velocity
    self.base_energy += 0.1
    self.energy += (self.base_energy - self.energy) * 0.01
    radius = 1 + self.energy / 50
    if self.pressed:
      self.velocity = -1
      self.updrafts.append(Updraft(self.y + radius))
    for updraft in self.updrafts:
      updraft.timer -= 1
    self.updrafts = [updraft for updraft in self.updrafts if updraft.timer > 0]

    reward = 0
    for fish in self.fish:
      fish.x -= 1
      horizontal_gap = abs(fish.x - EMBER_X) - fish.width / 2
      if horizontal_gap <= 3 and any(abs(fish.y - updraft.y - 2.5) <= 6 for updraft in self.updrafts):
        self.energy += 5
        reward += math.floor(self.energy / (100 if fish.burned else 10))
        fish.burned = True
      distance = math.hypot(max(0, horizontal_gap), max(0, abs(fish.y - self.y) - 3))
      self.terminal |= distance <= radius + 1.5
    self.fish = [fish for fish in self.fish if fish.x + fish.width >= 0]
    if self.fish[-1].x < 100:
      self.fish.append(Fish(self.fish[-1].x + self.random.uniform(50, 80),
                            self.random.uniform(10, 90), self.random.uniform(10, 20)))
    self.terminal |= not -radius <= self.y <= 100 + radius
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y) // 10))
    state[row, 2, 0] = True
    state[row, 2, 4] = self.velocity < 0
    for fish in self.fish:
      left = max(0, math.floor((fish.x - fish.width / 2) / 10))
      right = min(10, math.ceil((fish.x + fish.width / 2) / 10))
      if left < right:
        state[int(fish.y) // 10, left:right, 2 if fish.burned else 1] = True
    for updraft in self.updrafts:
      if 0 <= updraft.y < 100:
        state[int(updraft.y) // 10, 2, 3] = True
    for row, value in enumerate((self.steps, round(self.energy), round(self.velocity * 10) + 128)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
