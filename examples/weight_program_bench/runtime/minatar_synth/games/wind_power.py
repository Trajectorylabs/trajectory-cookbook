from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass
class Obstacle:
  x: float
  y: float
  size: float
  vx: float = 0.


class WindPowerEnvironment:
  score_cap = 1500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.wind = self.speed = self.angle = self.fan_angle = 0.
    self.obstacles = []
    self.timer = 0.
    self.action = self.steps = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0.
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self):
    self.wind = min(1, self.wind + 0.02) if self.action == 5 else self.wind * 0.9
    self.fan_angle = (self.fan_angle + self.wind * 0.4) % (2 * pi)
    self.speed = min(10, self.speed + self.wind * 0.02) * 0.98
    self.angle = (self.angle + self.speed * 0.2) % (2 * pi)
    self.timer -= 1
    if self.timer < 0:
      size = self.random.uniform(5, 10)
      self.obstacles.append(Obstacle(self.random.uniform(20, 99), -size, size))
      self.timer = self.random.uniform(75, 250)
    blades = list(self._blades())
    for obstacle in self.obstacles:
      obstacle.y += 0.5
      if 20 < obstacle.y < 100:
        obstacle.vx += self.wind / obstacle.size
      obstacle.vx *= 0.9
      obstacle.x += obstacle.vx
      radius = obstacle.size / 2 + 2
      if any(abs(x - obstacle.x) < radius and abs(y - obstacle.y) < radius for x, y in blades):
        self.terminal = True
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y <= 100]
    return self.speed

  def _blades(self):
    for index in range(3):
      angle = self.angle + index * 2 * pi / 3
      for radius in range(11):
        yield 85 + radius * cos(angle), 60 + radius * sin(angle)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[6, 1, 0] = True
    tip = 60 + sin(self.fan_angle) * 20
    state[int(min(60, tip) // 10):int(max(60, tip) // 10) + 1, 1, 0] = True
    for x, y in self._blades():
      state[int(y // 10), int(x // 10), 1] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for obstacle in self.obstacles:
      mask = ((abs(cols - obstacle.x) < obstacle.size / 2 + 5)
              & (abs(rows - obstacle.y) < obstacle.size / 2 + 5))
      state[:, :, 2] |= mask
      if 20 < obstacle.y < 100:
        state[:, :, 3] |= mask
    state[9, :round(self.wind * 10), 4] = True
    values = (self.steps, self.action, round(self.wind * 100), round(self.speed * 100),
              round(self.angle * 64), round(self.timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
