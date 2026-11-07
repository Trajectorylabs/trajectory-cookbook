from dataclasses import dataclass
from math import ceil, cos, floor, pi, sin

import numpy as np


SECTOR = 2 * pi / 32


@dataclass
class Bar:
  y: float
  grows_spikes: bool
  width: float = 0


class RWheelEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y = self.vy = self.angle = 0.0
    self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.spikes = [0] * 32
    self.hit = [False] * 32
    self.bars = []
    self.bonuses = []

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
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _add_bonus(self):
    for _ in range(ceil(sum(height > 0 for height in self.spikes) / 9)):
      angle = -2 * pi / 3 + self.random.uniform(-3 * pi / 4, 3 * pi / 4)
      self.bonuses.append((angle, self.random.uniform(10, 30)))

  def _advance(self, tap):
    if tap:
      self.vy = -2
      self.hit = [False] * 32
      if self.y == 0:
        self.y = self.vy
        self.multiplier = 0
        self._add_bonus()
    if self.y < 0:
      previous = self.vy
      self.vy = (self.vy + (0.03 if self.held else 0.09)) * 0.98
      if self.y < -74 and self.vy < 0:
        self.vy *= -0.5
      if previous * self.vy <= 0:
        self.bars.append(Bar(90 + self.y, True))
      self.y += self.vy
      if self.y > 0:
        self.y = self.vy = 0
    bonuses = []
    for angle, radius in self.bonuses:
      if abs(radius * cos(angle)) < 5 and abs(50 + radius * sin(angle) - (87 + self.y)) < 6:
        self.bars.append(Bar(90 + self.y, False))
      else:
        bonuses.append((angle + 0.03, radius))
    self.bonuses = bonuses
    for bar in self.bars:
      bar.width += 1 if bar.grows_spikes else 2
      bar.y += 3 if bar.grows_spikes else 2
    self.bars = [bar for bar in self.bars if bar.y <= 103]
    self.angle = (self.angle + 0.03) % (2 * pi)
    reward = 0
    for index, height in enumerate(self.spikes):
      xs, ys = self._spike_points(index, height)
      if not self.hit[index] and any(
        bar.grows_spikes and self._overlap(xs, ys, bar.y, bar.width / 2, 1.5)
        for bar in self.bars
      ):
        height += 1
        self.hit[index] = True
      if height:
        if any(not bar.grows_spikes and self._overlap(xs, ys, bar.y, bar.width / 2, 1.5)
               for bar in self.bars):
          self.multiplier += height
          reward += self.multiplier
          height = 0
        elif self._overlap(xs, ys, 87 + self.y, 3, 3):
          self.terminal = True
      self.spikes[index] = height
    if not any(self.spikes):
      for angle in (-pi / 4, -3 * pi / 4):
        self.spikes[floor((angle - self.angle) / SECTOR) % 32] = 1
      self._add_bonus()
    return reward

  def _spike_points(self, index, height):
    angle = self.angle + index * SECTOR
    dx, dy = cos(angle), sin(angle)
    radius = 40 - 4 * height
    fractions = np.linspace(0, 1, 2 + 4 * height)
    lengths = radius + (40 - radius) * fractions
    side = fractions * (50 / 32)
    return (
      np.concatenate((50 + dx * lengths - dy * side, 50 + dx * lengths + dy * side)),
      np.concatenate((50 + dy * lengths + dx * side, 50 + dy * lengths - dx * side)),
    )

  @staticmethod
  def _overlap(xs, ys, y, half_width, half_height):
    return bool(np.any((abs(xs - 50) <= half_width) & (abs(ys - y) <= half_height)))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int((87 + self.y) // 10), 5, 0] = True
    for index, height in enumerate(self.spikes):
      if height:
        x, y = self._spike_points(index, height)
        visible = (x >= 0) & (x < 100) & (y >= 0) & (y < 100)
        state[(y[visible] // 10).astype(int), (x[visible] // 10).astype(int), 1] = True
    for angle, radius in self.bonuses:
      state[int((50 + radius * sin(angle)) // 10),
            int((50 + radius * cos(angle)) // 10), 2] = True
    columns = np.arange(5, 100, 10)
    for bar in self.bars:
      if 0 <= bar.y < 100:
        state[int(bar.y // 10), :, 3 if bar.grows_spikes else 4] |= abs(columns - 50) < bar.width / 2 + 5
    for row, value in enumerate((self.steps, self.multiplier, int(self.held), round((self.vy + 2) * 100))):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
