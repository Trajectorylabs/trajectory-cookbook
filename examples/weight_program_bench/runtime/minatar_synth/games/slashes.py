from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Slash:
  x: float
  y: float
  slope: int
  width: float

  def overlaps(self, x, y, width, thickness, margin=0):
    dx, dy = x - self.x, y - self.y
    return ((abs(dx + self.slope * dy) < (width / 2 + margin) * sqrt(2))
            & (abs(dy - self.slope * dx) < (thickness / 2 + margin) * sqrt(2)))


class SlashesEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.slashes = [Slash(22, 50, 1, 30)]
    self.direction = 1
    self.speed = 1.
    self.pending = self.timer = 0.
    self.steps = 0
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.direction *= -1
    self.held = held
    self.steps += 1
    reward = 0.
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self):
    self.speed += (1 - self.speed) * 0.03
    self.timer -= 1
    if self.timer < 0:
      width = self.random.integers(40, 70)
      candidate = Slash(self.random.integers(-99, 199), -width,
                        self.random.choice((-1, 1)), width)
      points = np.linspace(-width / 2, width / 2, width) / sqrt(2)
      if not any(np.any(slash.overlaps(candidate.x + points,
                                       candidate.y + candidate.slope * points,
                                       slash.width, 4, 10)) for slash in self.slashes):
        candidate.width -= 20
        self.slashes.append(candidate)
      self.timer += self.random.uniform(10, 15)
    self.terminal = any(slash.overlaps(50, 90, slash.width, 4, 2) for slash in self.slashes)
    reward = 0.
    if not self.terminal:
      if any(slash.overlaps(50, 90, slash.width, 22, 2) for slash in self.slashes):
        self.speed += 0.03
        self.pending += self.speed
      else:
        reward, self.pending = self.pending, 0.
    for slash in self.slashes:
      slash.x += self.direction * self.speed
      slash.y += self.speed
    self.slashes = [slash for slash in self.slashes if slash.y < 99 + slash.width]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 5, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for slash in self.slashes:
      state[:, :, 1] |= slash.overlaps(cols, rows, slash.width, 4, 5)
      state[:, :, 2] |= slash.overlaps(cols, rows, slash.width, 22, 5)
    state[8, 4 if self.direction > 0 else 6, 3] = True
    state[9, 5, 4] = self.pending > 0
    for row, value in enumerate((self.steps, self.held, round(self.speed * 100),
                                 min(1023, round(self.pending * 4)))):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
