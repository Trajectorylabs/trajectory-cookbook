from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass
class Frog:
  x: float
  y: float
  angle: float


class RolfrgEnvironment:
  score_cap = 1600

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.direction = 50, 1
    self.frame = self.steps = 0
    self.held = self.terminal = False
    self.frogs = [Frog(0, -index * 9, self.random.uniform(0, 9)) for index in range(5)]
    self.bars = []
    self.long_bar = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      self._advance(tap and frame == 0)
      reward += 1
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    self.long_bar = self.frame % 30 == 0
    length = 30 if self.long_bar else 4
    self.bars = []
    for frog in self.frogs:
      if frog.y < 0:
        frog.x = self.random.uniform(0, 99)
        frog.y += self.random.uniform(99, 120)
      points = np.arange(length + 1)
      xs = frog.x + points * cos(frog.angle)
      ys = frog.y + points * sin(frog.angle)
      self.bars.append((xs, ys))
      frog.angle += 1 / 9
      frog.y -= 0.5
      if self.long_bar:
        frog.x += 30 * cos(frog.angle)
        frog.y += 30 * sin(frog.angle)
    self.x += self.direction
    if tap or self.x < 0 or self.x > 99:
      self.direction *= -1
    self.terminal = any(np.any((abs(xs - self.x) < 3) & (abs(ys - 36) < 3))
                        for xs, ys in self.bars)
    self.frame += 1

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[3, min(9, max(0, self.x // 10)), 0] = True
    for xs, ys in self.bars:
      visible = (xs >= 0) & (xs < 100) & (ys >= 0) & (ys < 100)
      state[(ys[visible] // 10).astype(int), (xs[visible] // 10).astype(int),
            2 if self.long_bar else 1] = True
    for frog in self.frogs:
      if 0 <= frog.x < 100 and 0 <= frog.y < 100:
        state[int(frog.y // 10), int(frog.x // 10), 3] = True
      x = frog.x + 12 * cos(frog.angle)
      y = frog.y + 12 * sin(frog.angle)
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 4] = True
    for row, value in enumerate((self.steps, self.frame % 30, self.direction > 0, self.held)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
