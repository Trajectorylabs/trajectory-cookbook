from dataclasses import dataclass
from math import pi, sin

import numpy as np


@dataclass
class Ground:
  x: float
  angle: float
  height: float = 9


@dataclass
class Enemy:
  x: float = 203
  y: float = 50
  vy: float = 0
  grounded: bool = True


class SShakeEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.ground = [Ground(i * 10, (i + 1) * pi / 5) for i in range(22)]
    self.terrain_x = np.arange(22) * 10
    self.terrain_y = np.full(22, 99.0)
    self.ratio = 1.0
    self.steps = self.timer = 0
    self.multiplier = 1
    self.held = self.terminal = False
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

  def _advance(self, tap):
    if tap:
      self.ratio = 3
      self.multiplier = 1
    self.ratio += (1 - self.ratio) * 0.05
    for index, ground in enumerate(self.ground):
      ground.x += 0.3
      if ground.x > 210:
        ground.x -= 220
        neighbor = self.ground[(index + 1) % 22]
        ground.angle = neighbor.angle - pi / 5 * self.random.uniform(0.5, 1.5)
        ground.height = 0.95 * (neighbor.height + self.random.uniform(-1, 1)) + 0.45
    ordered = sorted(self.ground, key=lambda ground: ground.x)
    self.terrain_x = np.array([ground.x for ground in ordered])
    heights = np.array([sin(ground.angle) * ground.height for ground in ordered])
    self.terrain_y = 99 + (heights - max(0, heights.max())) * self.ratio
    self.timer -= 1
    if self.timer < 0:
      self.enemies.append(Enemy())
      self.timer = self.random.uniform(0, 120)
    reward = 0
    remaining = []
    for enemy in self.enemies:
      enemy.x -= 0.3
      enemy.y += enemy.vy
      surface = float(np.interp(enemy.x, self.terrain_x, self.terrain_y))
      if enemy.grounded:
        if tap and enemy.y > surface - 2.5:
          launch = max(1, int(enemy.y - surface + 2.5))
          enemy.vy = -0.3 * launch
          enemy.grounded = False
        enemy.y = surface - 3.5
      else:
        enemy.vy += 0.03
        if abs(enemy.y - surface) <= 3.5 or (enemy.vy > 0 and surface < enemy.y <= surface + 30):
          enemy.y, enemy.vy, enemy.grounded = surface - 3.5, 0, True
      if enemy.y < -3:
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= enemy.x < 3
        if enemy.y <= 103:
          remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    columns = np.arange(10)
    surfaces = np.interp(columns * 20 + 10, self.terrain_x, self.terrain_y)
    state[(surfaces // 10).astype(int), columns, 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200 and 0 <= enemy.y < 100:
        row, col = int(enemy.y // 10), int(enemy.x // 20)
        channel = 1 if enemy.grounded else 2 if enemy.vy < 0 else 3
        state[row, col, channel] = True
    state[:, 0, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.held, round(self.ratio * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
