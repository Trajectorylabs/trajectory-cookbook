from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Falling:
  distance: float
  lane: int
  value: int = 0


class MeteoPlanetEnvironment:
  score_cap = 350

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.fallings = [Falling(53, int(self.random.integers(4)))]
    self.next_lane = int(self.random.integers(1, 4))
    self.angle = self.timer = 0.0
    self.target_lane = self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.target_lane += 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _position(self, falling):
    angle = falling.lane * math.pi / 2 - self.angle
    return 50 + math.cos(angle) * falling.distance, 50 + math.sin(angle) * falling.distance

  def _advance(self):
    self.angle = min(self.target_lane * math.pi / 2, self.angle + 0.3)
    self.timer -= 1
    if self.timer < 0:
      count = int(self.random.integers(6))
      for index in range(-count, count + 1):
        value = count - abs(index) + 1 if index else 0
        self.fallings.append(Falling(100 + 6 * index, self.next_lane, value))
      self.next_lane = (self.next_lane + int(self.random.integers(1, 4))) % 4
      self.timer = self.random.uniform(30, 50)
    reward = 0
    remaining = []
    for falling in self.fallings:
      falling.distance -= 0.5
      x, y = self._position(falling)
      extent = 3 + (falling.value / 2 if falling.value else 3)
      if abs(x - 50) < extent and abs(y - 42) < extent:
        if falling.value:
          reward += falling.value
          continue
        self.terminal = True
      if falling.distance >= 5:
        remaining.append(falling)
    self.fallings = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[4, 5, 0] = True
    state[5, 5, 1] = True
    for falling in self.fallings:
      x, y = self._position(falling)
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 2 if not falling.value else 3] = True
      if falling.distance < 25 and not falling.value and 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 4] = True
    gauges = (self.steps, self.target_lane % 4, round(self.angle % math.tau * 80))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
