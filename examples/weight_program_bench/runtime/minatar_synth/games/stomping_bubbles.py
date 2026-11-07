from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass
class Bubble:
  x: float
  y: float
  radius: float
  speed: float


@dataclass
class Chain:
  x: float
  y: float
  radius: float
  life: int = 5


class StompingBubblesEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 0.
    self.vx, self.vy = 1.5, 0.
    self.timer = self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.bubbles, self.chains = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.x += self.vx
    if (self.x > 95 and self.vx > 0) or (self.x < 5 and self.vx < 0):
      self.vx *= -1
    self.y += self.vy
    self.vy = (self.vy + (0.09 if self.held else 0.01)) * 0.99
    if self.y < 0 and self.vy < 0:
      self.vy *= -0.5
    self.terminal = self.y > 99
    self.timer -= 1
    if self.timer < 0:
      size = self.random.uniform(5, 9)
      self.bubbles.append(Bubble(self.random.uniform(0, 100), 102 + size,
                                 size, self.random.uniform(0.1, 0.3)))
      self.timer += 9
    for chain in self.chains:
      chain.radius += 1
      chain.life -= 1
    self.chains = [chain for chain in self.chains if chain.life > 0]
    active = tuple(self.chains)
    x, y = self.x, self.y
    reward = 0
    bounced = False
    remaining = []
    for bubble in self.bubbles:
      bubble.y -= bubble.speed
      dx, dy = abs(bubble.x - x), abs(bubble.y - y)
      near = hypot(max(0, dx - 2.5), max(0, dy - 2.5))
      far = hypot(dx + 2.5, dy + 2.5)
      direct = near < bubble.radius + 1 and far > bubble.radius - 1
      chained = any(self._intersects(bubble, chain) for chain in active)
      if direct or chained:
        reward += self.multiplier
        self.multiplier += 1
        self.chains.append(Chain(bubble.x, bubble.y, bubble.radius))
        if direct and not bounced:
          self.vy *= -1
          self.y += self.vy * 2
          bounced = True
      elif bubble.y >= -bubble.radius:
        remaining.append(bubble)
    self.bubbles = remaining
    if not self.chains:
      self.multiplier = 1
    return reward

  @staticmethod
  def _intersects(bubble, chain):
    distance = hypot(bubble.x - chain.x, bubble.y - chain.y)
    return abs(bubble.radius - chain.radius) - 2 < distance < bubble.radius + chain.radius + 2

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(np.clip(self.y // 10, 0, 9))
    state[row, int(self.x // 10), 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for channel, circles in ((1, self.bubbles), (2, self.chains)):
      for circle in circles:
        state[:, :, channel] |= abs(np.hypot(cols - circle.x, rows - circle.y) - circle.radius) < 7
    ahead = int(np.clip((self.x + 6 * self.vx) // 10, 0, 9))
    state[row, ahead, 3] = True
    state[9, :, 4] = True
    values = (self.steps, self.timer, self.multiplier, self.held,
              int(np.clip(round((self.vy + 15) * 16), 0, 511)))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
