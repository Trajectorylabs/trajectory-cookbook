from collections import deque
from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Enemy:
  x: float
  y: float
  z: float
  vy: float
  lock: int = -1


class TLaserEnvironment:
  score_cap = 500
  moves = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(400, horizon)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 50
    self.action = self.steps = self.timer = self.lock_index = 0
    self.multiplier = 1
    self.terminal = False
    self.enemies = []
    self.laser = None
    self.history = deque(maxlen=19)
    self.velocity = np.zeros(2)
    self.depth = 1.

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
    dx, dy = self.moves[self.action]
    self.x = min(99, max(0, self.x + dx * 3))
    self.y = min(99, max(0, self.y + dy * 3))
    target = min((enemy for enemy in self.enemies if enemy.lock >= 0),
                 key=lambda enemy: enemy.lock, default=None)
    if target is not None and self.laser is None:
      self.laser = np.array([50., 99.])
      self.velocity[:] = 0, -1
      self.depth = 1.
      self.multiplier = 1
      self.history.clear()
      self.history.append(self.laser.copy())
    if self.laser is not None:
      change = 0.
      if target is not None:
        self.velocity = (self.velocity + 0.5 * (np.array([target.x, target.y]) - self.laser)) * 0.5
        change = (target.z - self.depth) * 0.1
        self.depth += change
      else:
        speed = np.linalg.norm(self.velocity)
        if speed:
          self.velocity *= 5 / speed
      self.laser += self.velocity / (self.depth * (1 + 9 * abs(change)))
      self.history.append(self.laser.copy())
      margin = 99 / self.depth
      if np.any((self.laser < -margin) | (self.laser > 100 + margin)):
        self.laser = None
        self.history.clear()
        self.multiplier = 1
    self.timer -= 1
    if self.timer < 0:
      depth = self.random.uniform(0.5, 3)
      vy = self.random.choice((-1, 1)) * self.random.uniform(0.2, 0.3) / sqrt(depth)
      x, y = self.random.uniform(10, 90), -5 if vy > 0 else 105
      self.enemies.extend(Enemy(x, y - vy * i * 30, depth, vy)
                          for i in range(self.random.integers(3, 6)))
      self.timer += self.random.uniform(60, 99)
    reward = 0
    survivors = []
    for enemy in self.enemies:
      enemy.y += enemy.vy
      half = 3 / sqrt(enemy.z)
      if enemy is target and self._laser_hit(enemy, half):
        reward += self.multiplier
        self.multiplier += 1
        continue
      if enemy.lock < 0 and abs(enemy.x - self.x) < half + 3 and abs(enemy.y - self.y) < half + 3:
        enemy.lock = self.lock_index
        self.lock_index += 1
      if enemy.lock < 0 and (enemy.y < 0 if enemy.vy < 0 else enemy.y > 99):
        self.terminal = True
      survivors.append(enemy)
    self.enemies = survivors
    return reward

  def _laser_hit(self, enemy, half):
    if len(self.history) < 2:
      return False
    path = np.asarray(self.history)
    starts, vectors = path[:-1], np.diff(path, axis=0)
    delta = np.array([enemy.x, enemy.y]) - starts
    length_sq = np.sum(vectors * vectors, axis=1)
    ratio = np.sum(delta * vectors, axis=1) / np.maximum(length_sq, 1e-9)
    nearest = delta - vectors * np.clip(ratio, 0, 1)[:, None]
    return bool(np.any(np.linalg.norm(nearest, axis=1) < half + 1.5 / self.depth))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.y // 10, self.x // 10, 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.y < 100:
        col, row = int(enemy.x // 10), int(enemy.y // 10)
        state[row, col, 1 if enemy.lock < 0 else 2] = True
        state[row, col, 4] |= enemy.vy > 0
    for start, end in zip(self.history, list(self.history)[1:]):
      points = np.linspace(start, end, 20)
      points = points[np.all((points >= 0) & (points < 100), axis=1)]
      cols, rows = (points // 10).astype(int).T
      state[rows, cols, 3] = True
    for row, value in enumerate((self.steps, self.multiplier, self.action,
                                 max(0, round(self.timer)), round(self.depth * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
