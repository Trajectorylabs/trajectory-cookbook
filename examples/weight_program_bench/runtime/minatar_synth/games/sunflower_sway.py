from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass
class Falling:
  x: float
  y: float
  speed: float


class SunflowerSwayEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.height, self.angle, self.velocity = 40., 0., 0.001
    self.sun = 120.
    self.drop_timer, self.insect_timer = 30., 60.
    self.steps = 0
    self.held = self.terminal = False
    self.drops, self.insects = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0.
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _head(self):
    return 30 + self.height * sin(self.angle), 95 - self.height * cos(self.angle)

  def _advance(self):
    if self.held:
      self.velocity += 0.005
    self.sun += ((80 if self.held else 120) - self.sun) * 0.1
    self.velocity *= 0.97
    self.angle += self.velocity
    self.velocity -= self.angle * 0.01
    self.height = min(70, self.height - 0.045)
    self.terminal = self.height < 9
    x, y = self._head()
    self.drop_timer -= 1
    if self.drop_timer <= 0:
      self.drops.append(Falling(self.random.uniform(0, 99), -3, self.random.uniform(0.5, 1)))
      self.drop_timer = self.random.uniform(30, 40)
    reward = 0.
    remaining = []
    for drop in self.drops:
      drop.y += drop.speed
      if abs(drop.x - x) < 5.5 and abs(drop.y - y) < 5.5:
        reward += self.height
        self.height += 7
      elif drop.y <= 90:
        remaining.append(drop)
    self.drops = remaining
    self.insect_timer -= 1
    if self.insect_timer <= 0:
      bounds = (0, 20) if self.random.random() < 0.25 else (40, 99)
      self.insects.append(Falling(self.random.uniform(*bounds), -3, self.random.uniform(0.3, 0.8)))
      self.insect_timer = self.random.uniform(60, 70)
    remaining = []
    for insect in self.insects:
      insect.y += insect.speed
      if abs(insect.x - x) < 6 and abs(insect.y - y) < 6:
        self.height -= 15
      elif insect.y <= 90:
        remaining.append(insect)
    self.insects = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = self._head()
    row, col = np.clip(np.array([y, x]) // 10, 0, 9).astype(int)
    state[row, col, 0] = True
    stem = np.linspace((30, 95), (x, y), 24)
    cols, rows = np.clip(stem // 10, 0, 9).astype(int).T
    state[rows, cols, 1] = True
    for channel, objects in ((2, self.drops), (3, self.insects)):
      for obj in objects:
        if 0 <= obj.y < 100:
          state[int(obj.y // 10), int(obj.x // 10), channel] = True
    if self.sun < 100:
      state[1, int(self.sun // 10), 4] = True
    values = (self.steps, round(self.height * 5), self.held, round((self.angle + 2) * 100),
              round((self.velocity + 1) * 100), round(self.drop_timer), round(self.insect_timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
