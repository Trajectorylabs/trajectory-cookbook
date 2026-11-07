from dataclasses import dataclass
from math import floor

import numpy as np


@dataclass
class Diamond:
  x: float
  y: float
  size: float
  vx: float = 0.


class ThrustLrEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.vx, self.distance = 50., 0., 0.
    self.action = self.steps = 0
    self.terminal = False
    self.diamonds = []
    for y in (60, 40, 20):
      self._spawn(y)

  def _spawn(self, y=0):
    small = self.random.random() < 0.3
    size = self.random.uniform(7, 12) if small else self.random.uniform(20, 30)
    x = self.random.uniform(10, 90) if small else (
      50 + self.random.choice((-1, 1)) * self.random.uniform(40, 60)
    )
    self.diamonds.append(Diamond(x, y - size, size))
    return size

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    thrust = 50 * ((self.action == 3) - (self.action == 1))
    self.distance -= 0.2
    if self.distance < 0:
      size = self._spawn()
      self.distance = 10 + (size - 10) * self.random.random()
    reward = 0
    remaining = []
    for diamond in self.diamonds:
      diamond.y += 0.2
      dy = abs(diamond.y - 85)
      if dy < diamond.size:
        dx = diamond.x - self.x
        if dx * thrust < 0:
          clearance = abs(dx) - (diamond.size - dy)
          diamond.vx -= thrust * 0.2 / (max(0.5, clearance) * diamond.size)
        diamond.vx *= 0.98
      diamond.x += diamond.vx
      if diamond.x < -diamond.size or diamond.x > 99 + diamond.size:
        reward += floor(diamond.size)
      elif diamond.y - 2 * diamond.size <= 99:
        remaining.append(diamond)
    self.diamonds = remaining
    self.vx += thrust * 0.001
    self.x = min(96, max(4, self.x + self.vx))
    if (self.x < 5 and self.vx < 0) or (self.x > 95 and self.vx > 0):
      self.vx *= -0.7
    self.vx *= 0.95
    self.terminal = any(
      abs(abs(self.x + side * (1 + t) - diamond.x) + abs(83 + 4 * t - diamond.y)
          - diamond.size) < 2.5
      for diamond in self.diamonds for side in (-1, 1) for t in (0, 0.5, 1)
    )
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, int(self.x // 10), 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for diamond in self.diamonds:
      mask = abs(abs(cols - diamond.x) + abs(rows - diamond.y) - diamond.size) < 10
      state[:, :, 1 if diamond.size < 15 else 2] |= mask
      if abs(diamond.y - 85) < diamond.size:
        state[:, :, 3] |= mask
    if self.action in (1, 3):
      offset = 10 if self.action == 1 else -10
      state[8, int(np.clip((self.x + offset) // 10, 0, 9)), 4] = True
    values = (self.steps, self.action, round((self.vx + 2) * 100), round(self.distance * 10))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
