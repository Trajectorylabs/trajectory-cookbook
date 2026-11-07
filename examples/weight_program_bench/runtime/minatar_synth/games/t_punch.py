from dataclasses import dataclass
from math import cos, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Orb:
  x: float
  y: float
  vx: float
  vy: float
  age: int = 0


class TPunchEnvironment:
  score_cap = 1500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = self.angular_velocity = self.length = 0.
    self.arms = [True] * 6
    self.enemies, self.bonuses = [], []
    self.timer = self.steps = 0
    self.multiplier = 1
    self.attacking = self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.attacking = self.length < 5
      self.multiplier = 1
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _segments(self):
    size = 3 + sqrt(self.length) * 0.5
    for index, alive in enumerate(self.arms):
      if not alive:
        continue
      angle = self.angle + (index + 1) * pi / 3
      for segment in range(5):
        radius = 5 + segment * self.length / 4
        punch = segment == 4 and self.attacking
        yield index, 50 + cos(angle) * radius, 50 + sin(angle) * radius, size * (1.5 if punch else 1), punch

  @staticmethod
  def _contact(orb, segment):
    _, x, y, size, _ = segment
    return abs(orb.x - x) < 3 + size / 2 and abs(orb.y - y) < 3 + size / 2

  def _advance(self):
    if not self.held or self.length > 30:
      self.attacking = False
    self.length += ((36 if self.held else 0) - self.length) * 0.1
    self.angular_velocity += 0.001 if self.held else (0.03 - self.angular_velocity) * 0.2
    self.angle = (self.angle + self.angular_velocity) % (2 * pi)
    segments = list(self._segments())
    self.timer = self.timer - 1 if self.enemies else -1
    if self.timer < 0:
      x = self.random.uniform(0, 99)
      y = int(self.random.choice((-3, 3)))
      if self.random.random() < 0.5:
        x, y = y, x
      vx, vy = self.random.uniform(-0.3, 0.3, 2)
      self.enemies.append(Orb(x, y, vx, vy))
      self.timer = self.random.uniform(60, 99)
    reward = 0
    remaining = []
    for bonus in self.bonuses:
      distance = max(1e-9, hypot(bonus.x - 50, bonus.y - 50))
      bonus.vx = (bonus.vx + 0.002 * (bonus.x - 50) / distance) * 0.98
      bonus.vy = (bonus.vy + 0.002 * (bonus.y - 50) / distance) * 0.98
      bonus.x += bonus.vx
      bonus.y += bonus.vy
      if bonus.age > 60 and any(self._contact(bonus, segment) for segment in segments):
        reward += self.multiplier
        self.multiplier += 1
      elif -3 <= bonus.x <= 103 and -3 <= bonus.y <= 103:
        bonus.age += 1
        remaining.append(bonus)
    self.bonuses = remaining
    remaining = []
    for enemy in self.enemies:
      distance = hypot(50 - enemy.x, 50 - enemy.y)
      if distance > 30:
        enemy.vx = (enemy.vx + 0.005 * (50 - enemy.x) / distance) * 0.99
        enemy.vy = (enemy.vy + 0.005 * (50 - enemy.y) / distance) * 0.99
      enemy.x += enemy.vx
      enemy.y += enemy.vy
      hits = [segment for segment in segments if self._contact(enemy, segment)]
      if any(segment[-1] for segment in hits):
        self.bonuses.extend(Orb(enemy.x, enemy.y, i * 0.05 * cos(i), i * 0.05 * sin(i))
                            for i in range(16))
        start = int(self.random.integers(6))
        for offset in range(6):
          index = (start + offset) % 6
          if not self.arms[index]:
            self.arms[index] = True
            break
      elif hits:
        for index, *_ in hits:
          self.arms[index] = False
      else:
        self.terminal |= abs(enemy.x - 49) < 6 and abs(enemy.y - 49) < 6
        remaining.append(enemy)
    self.enemies = remaining
    self.terminal |= not any(self.arms)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[4, 4, 0] = True
    points = [(x, y, int(punch)) for _, x, y, _, punch in self._segments()]
    points.extend((enemy.x, enemy.y, 2) for enemy in self.enemies)
    points.extend((bonus.x, bonus.y, 4 if bonus.age > 60 else 3) for bonus in self.bonuses)
    for x, y, channel in points:
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), channel] = True
    values = (self.steps, self.multiplier, self.held, self.attacking,
              round(self.length * 10), round(self.angle * 70))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
