from dataclasses import dataclass
from math import sin

import numpy as np


@dataclass
class Obstacle:
  x: float
  y: float
  width: float
  height: float
  underwater: bool


class TurtleTideEnvironment:
  score_cap = 600

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y = self.water = 50.
    self.speed, self.frequency, self.amplitude = 1., 0.05, 10.
    self.obstacles = []
    self.obstacle_distance = 60.
    self.action = self.steps = self.frames = 0
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
    self.frequency = min(0.05, max(0, self.frequency + self.random.uniform(-0.0001, 0.0001)))
    self.amplitude = min(10, max(0, self.amplitude + self.random.uniform(-0.01, 0.01)))
    target_water = 50 + sin(self.frames * self.frequency) * self.amplitude
    self.water += (target_water - self.water) * 0.1
    self.frames += 1
    if self.action == 5:
      self.speed = min(9, self.speed + 0.01)
      self.y += (self.water + 15 - self.y) * 0.1
    else:
      self.speed += (1 - self.speed) * 0.1
      self.y += (self.water - 5 - self.y) * 0.1
      if self.y + 1 > self.water:
        self.y = self.water - 1

    for obstacle in self.obstacles:
      obstacle.x -= self.speed
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.x >= -10]
    self.obstacle_distance -= self.speed
    if self.obstacle_distance < 0:
      underwater = self.random.random() < 0.5
      low, high = (self.water + 5, 95) if underwater else (5, self.water - 5)
      self.obstacles.append(Obstacle(110, self.random.uniform(low, high),
                                     self.random.uniform(5, 10), self.random.uniform(5, 10),
                                     underwater))
      self.obstacle_distance = self.random.uniform(99, 120)
    self.terminal = any(
      abs(obstacle.x - 20) < obstacle.width / 2 + self.speed
      and abs(obstacle.y - self.y) < obstacle.height / 2 + 1
      for obstacle in self.obstacles
    )
    return self.speed * 0.1

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[int(self.y // 10), 2, 0] = True
    state[:, :, 1] = rows + 5 > self.water
    for obstacle in self.obstacles:
      mask = ((abs(cols - obstacle.x) < obstacle.width / 2 + 5)
              & (abs(rows - obstacle.y) < obstacle.height / 2 + 5))
      state[:, :, 3 if obstacle.underwater else 2] |= mask
    target = self.water + (15 if self.action == 5 else -5)
    state[int(target // 10), 2, 4] = True
    values = (self.steps, self.action, round(self.speed * 50), round(self.water * 8) - 256,
              round(self.obstacle_distance * 4), round(self.frequency * 10000),
              round(self.amplitude * 40))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
