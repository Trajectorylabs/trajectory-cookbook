from dataclasses import dataclass
from math import ceil, cos, hypot, pi, sin

import numpy as np


@dataclass
class Projectile:
  x: float
  y: float
  vx: float
  vy: float

  def move(self):
    self.x += self.vx
    self.y += self.vy


@dataclass
class Enemy:
  x: float
  y: float
  vx: float
  score: float = 9
  fired: bool = False


class RollHoldEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = 0.0
    self.direction = 1
    self.steps = self.fire_timer = self.enemy_timer = 0
    self.held = self.terminal = False
    self.shots, self.enemies, self.bullets = [], [], []
    self._schedule_enemy()

  def _schedule_enemy(self):
    vx = 0.4 if self.random.random() < 0.5 else -0.4
    self.next_enemy = (-3 if vx > 0 else 103, self.random.uniform(9, 90), vx)

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.direction *= -1
      self.fire_timer = 0
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    dx, dy = cos(self.angle), sin(self.angle)
    turret_x, turret_y = 50 + 12 * dx, 50 + 12 * dy
    if self.held:
      self.fire_timer -= 1
      if self.fire_timer < 0:
        self.shots.append(Projectile(turret_x, turret_y, 2 * dx, 2 * dy))
        self.fire_timer = 9
    else:
      self.angle = (self.angle + self.direction * 0.07 + pi) % (2 * pi) - pi
    for shot in self.shots:
      shot.move()
    self.shots = [shot for shot in self.shots if -9 <= shot.x <= 111 and -9 <= shot.y <= 111]
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      self.enemies.append(Enemy(*self.next_enemy))
      if self.random.random() < 0.25:
        self._schedule_enemy()
        self.enemy_timer = 120
      else:
        self.enemy_timer = 25
    reward = 0
    survivors = []
    for enemy in self.enemies:
      enemy.x += enemy.vx
      if not enemy.fired and (enemy.x > 90 if enemy.vx > 0 else enemy.x < 9):
        distance = hypot(50 - enemy.x, 50 - enemy.y)
        self.bullets.append(Projectile(enemy.x, enemy.y, 0.3 * (50 - enemy.x) / distance,
                                      0.3 * (50 - enemy.y) / distance))
        enemy.fired = True
      if abs(enemy.x - turret_x) < 5 and abs(enemy.y - turret_y) < 5:
        reward += 10
      elif any(abs(enemy.x - shot.x) < 4.5 and abs(enemy.y - shot.y) < 4.5 for shot in self.shots):
        reward += max(0, ceil(enemy.score))
      elif -5 <= enemy.x <= 105:
        enemy.score -= 0.033
        survivors.append(enemy)
    self.enemies = survivors
    bullets = []
    for bullet in self.bullets:
      bullet.move()
      if abs(bullet.x - turret_x) < 4 and abs(bullet.y - turret_y) < 4:
        reward += 10
      elif abs(bullet.x - 50) < 5 and abs(bullet.y - 50) < 5:
        self.terminal = True
      else:
        bullets.append(bullet)
    self.bullets = bullets
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    x, y = 50 + 12 * cos(self.angle), 50 + 12 * sin(self.angle)
    state[int(y // 10), int(x // 10), 1] = True
    for channel, entities in ((2, self.shots), (3, self.enemies), (4, self.bullets)):
      for entity in entities:
        if 0 <= entity.x < 100 and 0 <= entity.y < 100:
          state[int(entity.y // 10), int(entity.x // 10), channel] = True
    values = (self.steps, self.held, self.direction > 0, round((self.angle + pi) * 75),
              self.fire_timer, self.enemy_timer)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
