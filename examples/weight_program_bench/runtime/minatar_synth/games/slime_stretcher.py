from dataclasses import dataclass

import numpy as np


@dataclass
class Wall:
  x: float
  y: float
  width: float
  height: float

  def overlaps(self, x, y, width, height):
    return (x < self.x + self.width and x + width > self.x
            and y < self.y + self.height and y + height > self.y)


class SlimeStretcherEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 90.
    self.width = self.height = 20
    self.wall_distance, self.coin_distance = 50., 30.
    self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.walls, self.coins = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.held:
      self.height = min(80, self.height + 2)
      self.width = max(10, self.width - 0.5)
      self.y = max(10, self.y - 2)
      speed = 0.5
    else:
      self.height = max(20, self.height - 4)
      self.width = min(20, self.width + 1)
      speed = 1.5
    self.x += speed
    if self.x < 50:
      self.x += 1
    self.terminal = self.x < -self.width
    self.wall_distance -= speed
    if self.wall_distance <= 0:
      width = self.random.uniform(20, 60)
      self.walls.append(Wall(200, self.random.uniform(0, 70), width, self.random.uniform(20, 40)))
      self.wall_distance = width + self.random.integers(35, 55)
    self.coin_distance -= speed
    if self.coin_distance <= 0:
      self.coins.append((203, self.random.uniform(20, 80)))
      self.coin_distance = self.random.integers(30, 60)
    grounded = False
    for wall in self.walls:
      if not wall.overlaps(self.x, self.y, self.width, self.height):
        continue
      if self.y + self.height < wall.y + 10:
        self.height -= 1
        self.y = wall.y - self.height
        grounded = True
      elif self.y > wall.y + wall.height - 10:
        self.y = wall.y + wall.height
        self.height -= 2
      elif self.x + self.width < wall.x + 5:
        self.x = wall.x - self.width
        self.width -= 1
      else:
        self.y = wall.y - self.height
        grounded = True
    if not grounded:
      self.y = min(90, self.y + 1)
    self.y = min(self.y, 90 - self.height)
    self.x -= speed
    for wall in self.walls:
      wall.x -= speed
    self.walls = [wall for wall in self.walls if wall.x + wall.width >= 0]
    reward = 0
    remaining = []
    for x, y in self.coins:
      if any(wall.overlaps(x - 3, y - 3, 6, 6) for wall in self.walls):
        continue
      if self.x - 3 < x < self.x + self.width + 3 and self.y - 3 < y < self.y + self.height + 3:
        reward += self.multiplier
        self.multiplier += 1
      elif x - speed < -3:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append((x - speed, y))
    self.coins = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[:100:10, :200:20]
    state[:, :, 0] = (cols < self.x + self.width) & (cols + 20 > self.x) & (
      rows < self.y + self.height) & (rows + 10 > self.y)
    for wall in self.walls:
      state[:, :, 1] |= (cols < wall.x + wall.width) & (cols + 20 > wall.x) & (
        rows < wall.y + wall.height) & (rows + 10 > wall.y)
    for x, y in self.coins:
      if 0 <= x < 200:
        state[int(y // 10), int(x // 20), 2] = True
    state[9, :, 3] = True
    state[:, :, 4] = state[:, :, 0] & self.held
    values = (self.steps, self.multiplier, round(self.width * 10),
              round(self.height * 4), self.held)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
