from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass
class Shot:
  x: float
  y: float
  angle: float


@dataclass
class Enemy:
  x: float
  y: float = -3


class ScreenEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.target = 55.0
    self.angle = 0.0
    self.ammo = 3
    self.multiplier = 1
    self.steps = self.timer = 0
    self.held = self.terminal = False
    self.shots = []
    self.enemies = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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

  def _aim(self):
    return -self.angle if self.x > 50 else pi + self.angle

  def _advance(self, tap):
    self.angle = (self.angle + 0.01) % (pi / 2)
    if tap and self.ammo and abs(self.x - self.target) < 1:
      self.shots.append(Shot(self.x, 95, self._aim()))
      self.ammo -= 1
    if self.ammo == 0 and self.x == self.target:
      self.target = 100 - self.target
    previous = self.x
    self.x += (self.target - self.x) * 0.2
    if abs(self.target - self.x) < 1:
      self.x = self.target
    if (previous - 50) * (self.x - 50) <= 0:
      self.ammo = 3
    shots = []
    for shot in self.shots:
      shot.x += cos(shot.angle)
      shot.y += sin(shot.angle)
      if -5 <= shot.x <= 105 and -5 <= shot.y <= 105:
        shots.append(shot)
      else:
        self.multiplier = max(1, self.multiplier - 1)
    self.timer -= 1
    if self.timer < 0:
      x = 50 + self.random.choice((-1, 1)) * self.random.uniform(15, 40)
      self.enemies.append(Enemy(x))
      self.timer = self.random.uniform(60, 90)
    spent = set()
    remaining = []
    reward = 0
    for enemy in self.enemies:
      enemy.y += 0.1
      hits = [index for index, shot in enumerate(shots)
              if abs(shot.x - enemy.x) < 4 and abs(shot.y - enemy.y) < 4]
      if hits:
        spent.update(hits)
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= enemy.y > 99
        remaining.append(enemy)
    self.enemies = remaining
    self.shots = [shot for index, shot in enumerate(shots) if index not in spent]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x // 10), 0] = True
    for channel, entities in ((1, self.enemies), (2, self.shots)):
      for entity in entities:
        if 0 <= entity.x < 100 and 0 <= entity.y < 100:
          state[int(entity.y // 10), int(entity.x // 10), channel] = True
    distance = np.arange(100)
    x = self.x + cos(self._aim()) * distance
    y = 95 + sin(self._aim()) * distance
    visible = (x >= 0) & (x < 100) & (y >= 0) & (y < 100)
    state[(y[visible] // 10).astype(int), (x[visible] // 10).astype(int), 3] = True
    if self.x > 50:
      state[:, :5, 4] = True
    else:
      state[:, 5:, 4] = True
    for row, value in enumerate((self.steps, self.ammo, self.multiplier, self.held,
                                 round(self.angle * 100), self.target > 50)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
