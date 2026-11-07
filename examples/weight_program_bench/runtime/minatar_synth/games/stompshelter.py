from dataclasses import dataclass
from math import floor

import numpy as np


@dataclass
class Platform:
  x: float
  y: float
  width: float
  hp: int = 120


@dataclass
class Enemy:
  x: float
  y: float
  direction: int


class StompshelterEnvironment:
  score_cap = 600

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 70.
    self.vx, self.vy = 1, 0.
    self.steps = self.frame = self.jumps = self.combo = 0
    self.lava, self.distance = 102., 100.
    self.held = self.terminal = self.grounded = False
    self.platforms = [Platform(self.random.uniform(20, 60), y, self.random.uniform(25, 38))
                      for y in range(15, 91, 15)]
    self.enemies, self.debris = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
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

  def _advance(self, tap):
    self.lava -= 0.025
    top = min((platform.y for platform in self.platforms), default=15)
    if top > 5:
      self.platforms.append(Platform(self.random.uniform(10, 55), top - self.random.uniform(13, 17),
                                     self.random.uniform(25, 38)))
    if self.distance < 0:
      candidates = [p for p in self.platforms if p.y < 20 and p.hp > 60]
      if candidates:
        platform = candidates[self.random.integers(len(candidates))]
        self.enemies.append(Enemy(platform.x + platform.width / 2, platform.y - 5,
                                  int(self.random.choice((-1, 1)))))
        self.distance = self.random.uniform(100, 200)
    if self.frame % 70 == 0:
      self.debris.append((self.random.uniform(8, 92), -4))
    if tap and (self.grounded or self.jumps < 2):
      self.vy = -2.6
      self.jumps += 1
    self.vy += 0.1
    self.y += self.vy
    self.x += self.vx * (1 if self.held else 0.1)
    if (self.x > 90 and self.vx > 0) or (self.x < 10 and self.vx < 0):
      self.vx *= -1
    self.grounded = False
    for p in self.platforms:
      if self.vy > 0 and p.hp > 0 and p.x - 3 < self.x < p.x + p.width + 3 and p.y - 8 < self.y < p.y + 2:
        self.y, self.vy, self.grounded = p.y - 4, 0, True
        self.jumps = self.combo = 0
        p.hp -= 1
        break
    scroll = max(0, 60 - self.y)
    self.y += scroll
    self.lava += scroll
    self.distance -= scroll
    reward = floor(scroll)
    for p in self.platforms:
      p.y += scroll
    self.platforms = [p for p in self.platforms if p.y < 110 and p.hp > 0]
    self.lava -= max(0, self.lava - 100) * 0.5
    self.terminal = self.y > self.lava - 3
    remaining = []
    for enemy in self.enemies:
      enemy.y += scroll
      if enemy.y >= 105:
        continue
      enemy.x += enemy.direction * 0.35
      for p in self.platforms:
        if abs(enemy.y + 5 - p.y) < 2:
          if enemy.x <= p.x + 2:
            enemy.direction = 1
          if enemy.x >= p.x + p.width - 2:
            enemy.direction = -1
      if abs(enemy.x - self.x) < 6.5 and abs(enemy.y - self.y) < 5:
        if self.vy > 0.5 and self.y < enemy.y - 2:
          self.vy = -2.2
          self.combo += 1
          reward += 8 + self.combo * 4
          continue
        self.terminal = True
      remaining.append(enemy)
    self.enemies = remaining
    remaining = []
    for x, y in self.debris:
      y += scroll + 0.5
      if y >= 105:
        continue
      blockers = [p for p in self.platforms if p.hp > 0 and p.x < x < p.x + p.width and p.y - 2 < y < p.y + 5]
      if blockers:
        for p in blockers:
          p.hp -= 15
      else:
        self.terminal |= abs(x - self.x) < 5 and abs(y - self.y) < 4
        remaining.append((x, y))
    self.debris = remaining
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(self.x // 10), 0] = True
    rows, cols = np.ogrid[:100:10, :100:10]
    for p in self.platforms:
      if p.hp > 0:
        state[:, :, 1] |= (rows < p.y + 4) & (rows + 10 > p.y) & (cols < p.x + p.width) & (cols + 10 > p.x)
    for channel, positions in ((2, [(e.x, e.y) for e in self.enemies]), (3, self.debris)):
      for x, y in positions:
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    state[np.arange(10) * 10 + 10 > self.lava, :, 4] = True
    values = (self.steps, self.jumps, self.combo, self.held, self.vx > 0,
              int(np.clip(round((self.vy + 4) * 40), 0, 511)), self.grounded)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
