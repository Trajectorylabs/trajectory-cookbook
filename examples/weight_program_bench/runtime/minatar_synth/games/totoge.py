from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class SpikeStrip:
  x: float
  y: float
  count: int


class TotogeEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vx, self.vy = 50., 9., 0.33, 0.
    self.scroll, self.next_wall = 0., 105.
    self.spikes = []
    self.spike_distance, self.next_spike_x, self.spike_direction = 0., 10., 1
    self.action = self.steps = 0
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
    jump = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0.
    for frame in range(4):
      reward += self._advance(jump and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, jump):
    reward = 0.
    while self.next_wall < 105:
      self.next_wall += 10
      reward += 1
    self.next_wall -= self.scroll

    self.spike_distance -= self.scroll
    if self.spike_distance < 0:
      count = int(self.random.integers(2, 5))
      x = float(np.clip(self.next_spike_x, 12, 93 - count * 6))
      self.spikes.append(SpikeStrip(x, 199, count))
      self.spike_distance = self.random.uniform(99, 199)
      self.next_spike_x += self.spike_direction * count * 6
      if self.next_spike_x < 9 or self.next_spike_x > 90:
        self.spike_direction *= -1
    for strip in self.spikes:
      strip.y -= self.scroll
    self.spikes = [strip for strip in self.spikes if strip.y > -99]
    samples = ceil(min(99, max(1, self.vy / 6)))
    if any(self._hits(strip, self.y - 6 * offset)
           for strip in self.spikes for offset in range(samples)):
      self.terminal = True
      return reward

    self.x += self.vx
    self.y += self.vy - self.scroll
    if self.y < 10:
      self.scroll = (self.y - 10) / 2
    elif self.y > 30:
      self.scroll = (self.y - 30) / 2
    self.vy += 0.05
    if (self.x < 9 and self.vx < 0) or (self.x > 90 and self.vx > 0):
      self.vx *= -1
    if jump and self.vy > 0 and self.next_wall < 150:
      bonus = self.vy ** 2
      if bonus > 10:
        reward += bonus
      self.vy = -1
    return reward

  def _hits(self, strip, y):
    draw_y = min(99, max(-99, strip.y))
    for index in range(strip.count):
      dx = abs(self.x - strip.x - index * 6)
      dy = y - draw_y
      if dx < 6 and -5 < dy < 5 and dx < 4.5 + dy * 0.3:
        return True
    return False

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    col = int(np.clip(self.x // 10, 0, 9))
    row = int(np.clip(self.y // 10, 0, 9))
    state[row, col, 0] = True
    for strip in self.spikes:
      if 0 <= strip.y < 100:
        for index in range(strip.count):
          state[int(strip.y // 10), int((strip.x + index * 6) // 10), 1] = True
      elif strip.y >= 100:
        for index in range(strip.count):
          state[9, int((strip.x + index * 6) // 10), 2] = True
    for y in np.arange(self.next_wall - 10, -5, -10):
      if 0 <= y < 100:
        state[int(y // 10), (0, 9), 3] = True
    state[row, col, 4] = self.vy > 0
    values = (self.steps, self.action, self.vx > 0, round((self.vy + 1) * 16),
              round((self.scroll + 4) * 16), round(self.next_wall * 2))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
