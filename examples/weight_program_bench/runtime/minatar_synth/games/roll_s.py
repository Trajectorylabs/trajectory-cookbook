from dataclasses import dataclass
from math import atan2, cos, floor, pi, sin

import numpy as np


@dataclass
class Projectile:
  x: float
  y: float
  vx: float
  vy: float

  def advance(self, scroll):
    self.x += self.vx - scroll
    self.y += self.vy
    return -3 <= self.x <= 103 and -3 <= self.y <= 103


@dataclass
class Enemy:
  x: float
  y: int
  timer: int
  angle: float = 0.0


class RollSEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.aim = self.fire_timer = self.spawn_timer = self.steps = 0
    self.direction = self.multiplier = 1
    self.distance = 0.0
    self.held = self.terminal = False
    self.shots, self.enemies, self.bullets = [], [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    released = self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, released):
    if released:
      self.aim += self.direction
      if abs(self.aim) > 1:
        self.direction *= -1
        self.aim += self.direction * 2
      self.fire_timer = 9
    scroll = 0 if self.held else 0.5
    self.distance += scroll
    if self.distance > 100 * self.multiplier:
      self.multiplier += 1
    x, y = self._gun()
    if self.held:
      self.fire_timer -= 1
      if self.fire_timer < 0:
        for index in range(-2, 3):
          angle = self.aim * pi / 4 + index * 0.12
          self.shots.append(Projectile(x, y, 3 * cos(angle), 3 * sin(angle)))
        self.fire_timer = 9
    self.shots = [shot for shot in self.shots if shot.advance(scroll)]
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self.enemies.append(Enemy(105, 20 + 30 * self.random.integers(3), self.random.integers(300)))
      self.spawn_timer = self.random.uniform(50, 60)
    reward = 0
    enemies = []
    for enemy in self.enemies:
      gun_x, gun_y = enemy.x, enemy.y - 9
      enemy.timer -= 1
      if enemy.timer < 0:
        if enemy.timer % 36 == 0:
          self.bullets.append(Projectile(gun_x, gun_y, cos(enemy.angle), sin(enemy.angle)))
        if enemy.timer <= -108:
          enemy.timer = 300
      else:
        enemy.angle = floor((atan2(50 - enemy.y, 20 - enemy.x) + pi / 8) / (pi / 4)) * pi / 4
        enemy.x -= 0.2
      enemy.x -= scroll
      if any(self._near(shot.x, shot.y, gun_x, gun_y)
             or self._near(shot.x, shot.y, enemy.x, enemy.y - 3) for shot in self.shots):
        reward += self.multiplier
        continue
      self.terminal |= not (self.held and self.aim == 0) and self._near(x, y, gun_x, gun_y)
      if -5 <= enemy.x <= 105:
        enemies.append(enemy)
    self.enemies = enemies
    self.bullets = [bullet for bullet in self.bullets if bullet.advance(scroll)]
    self.terminal |= any(self._near(x, y, bullet.x, bullet.y) for bullet in self.bullets)
    return reward

  @staticmethod
  def _near(x, y, other_x, other_y):
    return abs(x - other_x) < 4 and abs(y - other_y) < 4

  def _gun(self):
    return (26, 47) if self.held and self.aim == 0 else (20, 41)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = self._gun()
    state[y // 10, x // 10, 0] = True
    for channel, entities in ((1, self.shots), (2, self.enemies), (4, self.bullets)):
      for entity in entities:
        if 0 <= entity.x < 100 and 0 <= entity.y < 100:
          row, col = int(entity.y // 10), int(entity.x // 10)
          state[row, col, channel] = True
          if channel == 2 and entity.timer < 0:
            state[row, col, 3] = True
    values = (self.steps, self.aim + 1, self.direction > 0, self.held,
              self.multiplier, self.fire_timer)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
