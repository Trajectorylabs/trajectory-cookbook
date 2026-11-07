from dataclasses import dataclass
from math import cos, pi

import numpy as np


@dataclass
class MovingObject:
  x: float
  y: float
  speed: float


class VineClimberEnvironment:
  score_cap = 30

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.vine_tick = 0
    self.amplitude, self.frequency, self.bend = 30., 0.03, 0.5
    self.target = (self.amplitude, self.frequency, self.bend)
    self.x, self.y = 50., 80.
    self.obstacles = []
    self.timer = 0.
    self.coin = None
    self.multiplier = 1
    self.steps = self.action = 0
    self.terminal = False

  def _vine_x(self, y):
    return 50 - cos((self.vine_tick + y * self.bend) * self.frequency) * self.amplitude

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
    self.y = min(99, max(0, self.y + (-0.8 if self.action == 5 else 1.2)))
    self.x = self._vine_x(self.y)
    self.vine_tick += 1
    amplitude, frequency, bend = self.target
    self.amplitude += (amplitude - self.amplitude) * 0.01
    self.frequency += (frequency - self.frequency) * 0.01
    self.bend += (bend - self.bend) * 0.01
    if self.random.random() < 0.005:
      self.target = tuple(self.random.uniform((25, 0.025, 0.45), (40, 0.04, 0.6)))
    if self.vine_tick * self.frequency >= 2 * pi:
      self.vine_tick = 0

    for obstacle in self.obstacles:
      obstacle.speed += 0.01
      obstacle.y += obstacle.speed
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y <= 105]
    self.terminal = any(self._touches(obstacle) for obstacle in self.obstacles)
    self.timer -= 1
    if self.timer < 0:
      self.obstacles.append(MovingObject(self.random.uniform(10, 90), 0,
                                         self.random.uniform(0, 0.5)))
      self.timer = self.random.uniform(70, 90)

    if self.coin is None:
      speed = self.random.choice((-1, 1)) * self.random.uniform(0.18, 0.25)
      self.coin = MovingObject(-10 if speed > 0 else 110, self.random.uniform(10, 90), speed)
    self.coin.x += self.coin.speed
    if self._touches(self.coin):
      reward = self.multiplier
      self.multiplier = min(9, self.multiplier + 1)
      self.coin = None
      return reward
    if not -10 <= self.coin.x <= 110:
      self.coin = None
      self.multiplier = 1
    return 0

  def _touches(self, obj):
    return abs(obj.x - self.x) < 5 and abs(obj.y - self.y) < 5

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), int(self.x // 10), 0] = True
    for y in range(0, 100, 2):
      state[y // 10, int(self._vine_x(y) // 10), 1] = True
    for obstacle in self.obstacles:
      if obstacle.y < 100:
        state[int(obstacle.y // 10), int(obstacle.x // 10), 2] = True
    if self.coin is not None:
      col = int(np.clip(self.coin.x // 10, 0, 9))
      state[int(self.coin.y // 10), col, 3] = True
      state[int(self.coin.y // 10), col, 4] = self.coin.speed > 0
    values = (self.steps, self.action, self.multiplier, self.vine_tick,
              round(self.amplitude * 10), round(self.frequency * 10000),
              round(self.bend * 100), round(self.timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
