from dataclasses import dataclass
from math import ceil, cos, hypot, pi, sin, sqrt

import numpy as np


_LEFT, _RIGHT = -4 * pi / 5, -pi / 5


@dataclass
class Drop:
  x: float
  y: float
  size: int = 2


class WiperEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle, self.direction = _LEFT, 1
    self.drops = []
    self.timer = self.action = self.steps = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    swing = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(swing and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _target(self):
    if self.angle > -pi / 3:
      return _LEFT
    if self.angle < -2 * pi / 3:
      return _RIGHT
    return _RIGHT if self.direction > 0 else _LEFT

  def _advance(self, swing):
    self.angle += self.direction * 0.05
    if (self.angle > _RIGHT and self.direction > 0) or (self.angle < _LEFT and self.direction < 0):
      self.direction *= -1
    if swing:
      self.angle = self._target()
      self.direction *= -1
    self.timer -= 1
    if self.timer < 0:
      self.drops.append(Drop(self.random.uniform(10, 90), self.random.uniform(0, 20)))
      self.timer = 10

    dx, dy = cos(self.angle), sin(self.angle)
    reward = 0
    live = dict(enumerate(self.drops))
    for index, drop in enumerate(self.drops):
      if index not in live:
        continue
      before = _blade_distance(drop, dx, dy)
      absorbed = False
      for other in live.values():
        if other.y < drop.y and hypot(drop.x - other.x, drop.y - other.y) < 2 * sqrt(drop.size + other.size):
          other.size += drop.size
          other.x = (other.x + drop.x) / 2
          absorbed = True
      if absorbed:
        del live[index]
        continue
      drop.y += sqrt(drop.size - 1.9) * 0.3
      if before <= 3 + sqrt(drop.size) or _blade_distance(drop, dx, dy) <= 3 + sqrt(drop.size) / 2:
        reward += ceil(drop.size - 1)
        del live[index]
      elif drop.y > 99:
        self.terminal = True
    self.drops = list(live.values())
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for drop in self.drops:
      col = int(drop.x // 10)
      row = min(9, int(drop.y // 10))
      state[row, col, 0 if drop.size == 2 else 1] = True
      if drop.y >= 85:
        state[row, col, 4] = True
    for channel, angle in ((2, self.angle), (3, self._target())):
      for radius in range(25, 62):
        x, y = 50 + radius * cos(angle), 110 + radius * sin(angle)
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    values = (self.steps, self.action, self.direction > 0, self.timer, round((self.angle + pi) * 128))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _blade_distance(drop, dx, dy):
  x, y = drop.x - 50, drop.y - 110
  along = min(61.2, max(25.2, x * dx + y * dy))
  return hypot(x - along * dx, y - along * dy)
