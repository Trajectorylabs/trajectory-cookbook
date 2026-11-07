from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Floor:
  x: float
  y: float
  width: float
  height: float


@dataclass
class Fan:
  x: float
  y: float
  speed: float
  angle: float


class SkyFloorEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 90.
    self.vx = self.vy = self.distance = 0.
    self.radius = 2.
    self.steps = self.action = 0
    self.multiplier = 1
    self.outside = self.terminal = False
    self.floors = [Floor(50, 50, 80, 150)]
    self.fans, self.coins = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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

  def _spawn(self):
    last = self.floors[-1]
    for _ in range(9):
      x, width = self.random.uniform(0, 99), self.random.uniform(30, 80)
      if abs(x - last.x) < (width + last.width) / 2:
        break
    else:
      x, width = last.x, last.width
    height = self.random.uniform(30, 99)
    y = -height / 2 - 20
    self.floors.append(Floor(x, y, width, height))
    for _ in range(self.random.integers(ceil(height / 30))):
      side = self.random.choice((-1, 1))
      self.fans.append(Fan(x + side * width * self.random.uniform(0.6, 0.7),
                           y + height * self.random.uniform(-0.4, 0.4),
                           self.random.uniform(1, 2),
                           (0 if side < 0 else pi) + self.random.uniform(-pi / 8, pi / 8)))
    count = self.random.integers(ceil(height / 10))
    start = self.random.uniform(-0.5, 0.5) * height - count * 5
    coin_x = x + width * self.random.uniform(-0.2, 0.2)
    for i in range(count):
      offset = start + i * 10
      if abs(offset) < height * 0.3:
        self.coins.append((coin_x, y + offset))
    self.distance = self.random.uniform(0.6, 0.8) * height

  def _advance(self):
    sx = (min(55, max(45, self.x)) - self.x) * 0.1
    sy = max(0, 80 - self.y) * 0.1
    self.distance -= sy
    if self.distance < 0:
      self._spawn()
    for floor in self.floors:
      floor.x += sx
      floor.y += sy
    self.floors = [floor for floor in self.floors if floor.y - floor.height / 2 <= 99]
    for fan in self.fans:
      fan.x += sx
      fan.y += sy
      dx, dy = self.x - fan.x, self.y - fan.y
      offset = abs((atan2(dy, dx) - fan.angle + pi) % (2 * pi) - pi)
      force = 3 * fan.speed * max(0, pi / 4 - offset) / (hypot(dx, dy) + 1)
      self.vx += force * cos(fan.angle)
      self.vy += force * sin(fan.angle)
    self.fans = [fan for fan in self.fans if fan.y <= 105]
    offset = 35 * ((self.action == 3) - (self.action == 1))
    self.vx = (self.vx + offset * abs(offset) * 0.0001) * 0.95
    self.vy = (self.vy - 0.1) * 0.95 * (1 - 0.03 * sqrt(abs(offset)))
    self.x += self.vx + sx
    self.y += self.vy + sy - 0.1
    if any(abs(floor.x - self.x) < floor.width / 2 + self.radius and
           abs(floor.y - self.y) < floor.height / 2 + self.radius for floor in self.floors):
      self.outside = False
      self.radius += (2 - self.radius) * 0.1
    else:
      if not self.outside:
        self.vx *= 0.5
      self.outside = True
      self.radius *= 0.925
      self.terminal = self.radius < 0.1
    reward = 0
    remaining = []
    for x, y in self.coins:
      x, y = x + sx, y + sy
      if abs(x - self.x) < 4 + self.radius and abs(y - self.y) < 4 + self.radius:
        reward += self.multiplier
        self.multiplier += 1
      elif y > 105:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append((x, y))
    self.coins = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = np.clip(np.array([self.y, self.x]) // 10, 0, 9).astype(int)
    state[row, col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for floor in self.floors:
      state[:, :, 1] |= (abs(rows - floor.y) < floor.height / 2 + 5) & (
        abs(cols - floor.x) < floor.width / 2 + 5)
    for channel, points in ((2, self.coins), (3, [(fan.x, fan.y) for fan in self.fans])):
      for x, y in points:
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    state[row, col, 4] = self.outside
    values = (self.steps, self.action, self.multiplier, round(self.radius * 100),
              round((self.vx + 20) * 20), round((self.vy + 20) * 20))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
