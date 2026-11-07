from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Magnet:
  x: float
  y: float = 0


class MagneticPendulumEnvironment:
  score_cap = 162

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 71.0
    self.length, self.angle, self.angular_velocity = 66.0, 0.0, 0.024
    self.floor = self.target_floor = 100.0
    self.spawn_timer = 70
    self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.magnets: list[Magnet] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.target_floor = min(100, self.target_floor + 0.01)
    self.floor += (self.target_floor - self.floor) * 0.5
    self.length = min(66, max(10, self.length + (-1 if self.pressed else 1)))
    self.spawn_timer -= 1
    if self.spawn_timer == 0:
      self.magnets.append(Magnet(self.random.uniform(15, 85)))
      self.spawn_timer = 70
    sine, cosine = math.sin(self.angle), math.cos(self.angle)
    tangent = 0.0
    for magnet in self.magnets:
      dx, dy = magnet.x - self.x, magnet.y - self.y
      distance = math.hypot(dx, dy)
      if 0 < distance < 30:
        tangent += 0.15 * (1 - distance / 30) * (dx * cosine - dy * sine) / distance
    self.angular_velocity = (self.angular_velocity + (tangent - 0.13 * sine) / self.length) * 0.99
    self.angle += self.angular_velocity
    self.x = 50 + math.sin(self.angle) * self.length
    self.y = 5 + math.cos(self.angle) * self.length
    self.terminal = self.y + 3 > self.floor
    if self.terminal:
      return 0
    reward = 0
    remaining = []
    for magnet in self.magnets:
      magnet.y += 0.45
      if abs(magnet.x - self.x) < 6 and abs(magnet.y - self.y) < 6:
        reward += self.multiplier
        self.multiplier = min(9, self.multiplier + 1)
        self.target_floor = min(100, self.target_floor + 2)
      elif magnet.y + 4 > self.floor:
        self.target_floor = max(0, self.target_floor - 10)
        self.multiplier = 1
      else:
        remaining.append(magnet)
    self.magnets = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for fraction in np.linspace(0, 1, 16):
      x = 50 + fraction * (self.x - 50)
      y = 5 + fraction * (self.y - 5)
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 1] = True
    state[min(9, max(0, int(self.y // 10))), min(9, max(0, int(self.x // 10))), 0] = True
    for magnet in self.magnets:
      row, col = int(magnet.y // 10), int(magnet.x // 10)
      state[row, col, 2] = True
      state[row, col, 3] |= math.hypot(magnet.x - self.x, magnet.y - self.y) < 30
    state[math.floor(self.floor / 10):, :, 4] = True
    values = (self.steps, int(self.length), self.multiplier, round(self.floor * 4),
              round((self.angular_velocity + 0.25) * 1000))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
