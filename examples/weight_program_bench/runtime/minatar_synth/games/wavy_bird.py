from dataclasses import dataclass
from math import cos, floor, sin

import numpy as np


@dataclass
class Wave:
  y: float
  angle: float
  radius: int = 0

  def points(self):
    for offset in (-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3):
      angle = self.angle + offset
      yield 20 + self.radius * cos(angle), self.y + self.radius * sin(angle)


class WavyBirdEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.vy, self.angle = 50., 0., 0.
    self.blocks, self.waves = [], []
    self.distance = 0.
    self.multiplier = 1.
    self.steps = self.action = 0
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
    flap = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(flap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, flap):
    points = []
    for wave in self.waves:
      wave.radius += 2
      points.extend(wave.points())
    self.waves = [wave for wave in self.waves if wave.radius < 30]
    self.y += self.vy
    self.vy += 0.1
    self.angle += (1.2 - self.angle) * 0.02
    if flap:
      self.waves.append(Wave(self.y, self.angle - 0.4))
      self.vy = -1.5
      self.angle -= 0.6
    self.angle = min(1.2, max(-1.2, self.angle))
    self.terminal = not 0 <= self.y <= 99

    reward = 0
    remaining = []
    bird_points = [(20 + offset * cos(self.angle), self.y + offset * sin(self.angle))
                   for offset in (-2.5, 0, 2.5)]
    for x, y in self.blocks:
      x -= 1
      if any(abs(x - wx) <= 5.5 and abs(y - wy) <= 5.5 for wx, wy in points):
        reward += floor(self.multiplier)
        self.multiplier += 1
        continue
      if any(abs(x - bx) <= 5.5 and abs(y - by) <= 5.5 for bx, by in bird_points):
        self.terminal = True
      if x >= -8:
        remaining.append((x, y))
    self.blocks = remaining

    self.distance -= 1
    if self.distance < 0:
      height = self.random.uniform(20, 60)
      y = self.random.uniform(0, 110 - height)
      self.blocks.extend((108, y + 8 * index) for index in range(int(height // 8)))
      self.distance += self.random.uniform(10, 40)
    self.multiplier = min(99, max(1, self.multiplier - 0.03))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(np.clip(self.y // 10, 0, 9))
    state[row, 2, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for x, y in self.blocks:
      state[:, :, 1] |= (abs(cols - x) < 9) & (abs(rows - y) < 9)
    for wave in self.waves:
      for x, y in wave.points():
        if 0 <= y < 100:
          state[int(y // 10), int(x // 10), 2] = True
    state[row, 2, 3] = self.vy < 0
    for x, y in Wave(self.y, self.angle - 0.4, 28).points():
      if 0 <= y < 100:
        state[int(y // 10), int(x // 10), 4] = True
    values = (self.steps, self.action, round((self.vy + 2) * 32),
              round((self.angle + 1.2) * 128), round(self.multiplier * 4),
              round(self.distance * 8))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
