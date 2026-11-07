from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass
class Shot:
  x: float
  y: float
  vx: float = 0.5
  vy: float = 0.


@dataclass
class Enemy:
  x: float
  y: float
  vy: float
  timer: float
  interval: float
  falling: bool = False


class ThrowMEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.vy = 50., -0.5
    self.angle = None
    self.steps = self.timer = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.enemies, self.bullets, self.shots = [], [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    press, release = held and not self.held, self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(press and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _hit(self, x, y):
    return any(abs(shot.x - x) < 5 and abs(shot.y - y) < 5 for shot in self.shots)

  def _advance(self, press, release):
    for shot in self.shots:
      shot.x += shot.vx
      shot.y += shot.vy
      shot.vy += 0.07
    self.shots = [shot for shot in self.shots if shot.y <= 103]
    self.timer -= 1
    if self.timer < 0:
      interval = self.random.uniform(200, 300)
      self.enemies.append(Enemy(self.random.uniform(3, 50), -5, self.random.uniform(0.1, 0.4),
                                self.random.uniform(0, interval), interval))
      self.timer = self.random.uniform(50, 60)
    reward = 0
    for enemy in self.enemies:
      enemy.y += enemy.vy
      if enemy.falling:
        enemy.vy += 0.1
        continue
      enemy.timer -= 1
      if enemy.timer < 0:
        self.bullets.append(Shot(enemy.x, enemy.y))
        enemy.timer = enemy.interval
      if self._hit(enemy.x, enemy.y) or self._hit(enemy.x, enemy.y - 6):
        enemy.falling = True
        reward += self.multiplier
        self.multiplier *= 2
    self.enemies = [enemy for enemy in self.enemies if enemy.y <= 105]
    bullets = []
    for bullet in self.bullets:
      bullet.x += bullet.vx
      if self._hit(bullet.x, bullet.y):
        reward += self.multiplier
        self.multiplier += 1
      elif bullet.x <= 103 and not (bullet.x > 83 and (bullet.y < 12 or bullet.y > 90)):
        bullets.append(bullet)
    self.bullets = bullets
    self.y += self.vy
    if (self.y < 19 and self.vy < 0) or (self.y > 90 and self.vy > 0):
      self.vy *= -1
    self.terminal = any(abs(bullet.x - 90) < 5 and abs(bullet.y - self.y + 6) < 4
                        for bullet in bullets)
    if self.angle is None and press:
      self.angle = 3 * pi / 4
    if self.angle is not None:
      self.angle += 0.1
      if release or self.angle > 11 * pi / 8:
        self.shots.append(Shot(90, self.y, 3 * cos(self.angle), 3 * sin(self.angle)))
        self.angle = None
        self.multiplier = 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), 9, 0] = True
    for channel, objects in ((1, self.enemies), (3, self.bullets), (4, self.shots)):
      for obj in objects:
        if 0 <= obj.x < 100 and 0 <= obj.y < 100:
          layer = 2 if channel == 1 and obj.falling else channel
          state[int(obj.y // 10), int(obj.x // 10), layer] = True
    values = (self.steps, self.held, self.vy > 0, round((self.angle or 0) * 100),
              min(1023, self.multiplier))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
