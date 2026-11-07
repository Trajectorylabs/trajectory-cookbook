from dataclasses import dataclass
from math import atan2, cos, floor, hypot, sin, sqrt

import numpy as np


@dataclass
class Debris:
  x: float
  y: float
  size: int
  distance: float = 0.
  angle: float = 0.


@dataclass
class Obstacle:
  x: float
  y: float
  width: float
  height: float


class TornadoTwisterEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.direction, self.speed = 75., 1, 1.
    self.size = self.current_size = 7.
    points = self.random.uniform(-10, 10, (20, 2))
    self.radii = np.hypot(points[:, 0], points[:, 1])
    self.angles = np.arctan2(points[:, 1], points[:, 0])
    self.debris, self.obstacles = [], []
    self.debris_timer = self.obstacle_timer = 0.
    self.steps = self.frames = self.action = 0
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
    if self.action == 5 and previous != 5:
      self.direction *= -1
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _cloud(self):
    angles = self.angles + self.frames * 0.15 * (1 - np.arange(20) / 20)
    radius = self.radii * self.current_size / 10
    return 20 + radius * np.cos(angles), self.y + radius * np.sin(angles)

  def _advance(self):
    if self.action == 5:
      self.current_size += (3 - self.current_size) * 0.1
      self.speed += (0.1 - self.speed) * 0.1
    else:
      self.current_size += (self.size - self.current_size) * 0.03
      self.speed += (1 - self.speed) * 0.1
    self.y += self.direction * self.speed
    if self.y <= self.current_size:
      self.direction = 1
    if self.y >= 100 - self.current_size:
      self.direction = -1

    self.frames += 1
    cloud_x, cloud_y = self._cloud()
    padding = sqrt(self.current_size)

    def touches(x, y, width, height):
      return np.any((abs(cloud_x - x) <= (width + padding) / 2)
                    & (abs(cloud_y - y) <= (height + padding) / 2))

    self.debris_timer -= 1
    if self.debris_timer <= 0:
      size = 3 if self.random.random() < 0.5 else (5 if self.random.random() < 0.7 else 4)
      self.debris.append(Debris(160, self.random.uniform(10, 90), size))
      self.debris_timer = self.random.uniform(30, 50)
    reward = 0
    remaining = []
    for item in self.debris:
      if item.distance > 0:
        item.distance -= 1
        item.angle += 0.2
        if item.distance <= 0:
          self.size += item.size * 0.05
          reward += floor(item.size * self.current_size)
          continue
        item.x = 20 + item.distance * cos(item.angle)
        item.y = self.y + item.distance * sin(item.angle)
      else:
        item.x -= 1
        if touches(item.x, item.y, item.size, item.size):
          dx, dy = item.x - 20, item.y - self.y
          item.distance = max(1, hypot(dx, dy) * 1.5)
          item.angle = atan2(dy, dx)
      if item.x >= -item.size:
        remaining.append(item)
    self.debris = remaining

    self.obstacle_timer -= 1
    if self.obstacle_timer < 0:
      width = self.random.uniform(20, 30)
      self.obstacles.append(Obstacle(150 + width / 2, self.random.uniform(10, 90),
                                     width, self.random.uniform(10, 20)))
      self.obstacle_timer += self.random.uniform(50, 150)
    for obstacle in self.obstacles:
      obstacle.x -= 1
    self.terminal = any(touches(o.x, o.y, o.width, o.height) for o in self.obstacles)
    self.obstacles = [o for o in self.obstacles if o.x > -o.width]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    cloud_x, cloud_y = self._cloud()
    for x, y in zip(cloud_x, cloud_y, strict=True):
      if 0 <= x < 150 and 0 <= y < 100:
        state[int(y // 10), int(x // 15), 0] = True
    for item in self.debris:
      if 0 <= item.x < 150 and 0 <= item.y < 100:
        state[int(item.y // 10), int(item.x // 15), 3 if item.distance else 1] = True
        if item.size >= 4 and not item.distance:
          state[int(item.y // 10), int(item.x // 15), 2] = True
    rows, cols = np.ogrid[0:100:10, 0:150:15]
    for obstacle in self.obstacles:
      state[:, :, 4] |= ((abs(cols + 7.5 - obstacle.x) < obstacle.width / 2 + 7.5)
                        & (abs(rows + 5 - obstacle.y) < obstacle.height / 2 + 5))
    values = (self.steps, self.action, self.direction > 0, round(self.speed * 100),
              round(self.current_size * 16), round(self.size * 16))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
