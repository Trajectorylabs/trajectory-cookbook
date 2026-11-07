from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass
class Tank:
  x: float
  y: float
  angle: float
  timer: float


@dataclass
class Bullet:
  x: float
  y: float
  vx: float
  vy: float


class UnctrlEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.tanks, self.bullets = [], []
    self.timer = 0.
    self.steps = self.action = 0
    self.terminal = False
    self._reload()

  def _reload(self):
    self.shot = Bullet(8, 50, 0.5, 0.2)
    self.fired = False
    self.multiplier = 1

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    fire = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(fire and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, fire):
    if self.fired:
      self.shot.vy += -0.1 if self.action == 5 else 0.1
      self.shot.x += self.shot.vx
      self.shot.y += self.shot.vy
      if not _inside(self.shot):
        self._reload()
    elif fire:
      self.fired = True

    self.timer = self.timer - 1 if self.tanks else -1
    if self.timer < 0:
      y = 50 + self.random.choice((-1, 1)) * self.random.uniform(20, 40)
      self.tanks.append(Tank(153, y, self._angle(), self.random.uniform(0, 60)))
      self.timer = 200

    reward = 0
    tanks = []
    for tank in self.tanks:
      tank.x -= 0.4 if tank.x < 70 else 0.1
      if self._hit(tank, 5):
        reward += self.multiplier * 10
        self._reload()
        continue
      tank.timer -= 1
      if tank.x > 70 and tank.timer < 0:
        self.bullets.append(Bullet(tank.x - 3, tank.y, cos(tank.angle) * 0.3,
                                   sin(tank.angle) * 0.3))
        tank.timer = 60
        tank.angle = self._angle()
      if tank.x >= -3:
        tanks.append(tank)
    self.tanks = tanks

    bullets = []
    for bullet in self.bullets:
      bullet.x += bullet.vx
      bullet.y += bullet.vy
      if self._hit(bullet, 4):
        reward += self.multiplier
        self.multiplier = min(64, self.multiplier * 2)
      else:
        if abs(bullet.x - 10) <= 4 and abs(bullet.y - 50) <= 4:
          self.terminal = True
        if _inside(bullet):
          bullets.append(bullet)
    self.bullets = bullets
    return reward

  def _hit(self, target, radius):
    return (self.fired and abs(self.shot.x + 1.5 - target.x) <= radius
            and abs(self.shot.y - target.y) <= radius)

  def _angle(self):
    return pi + self.random.uniform(-0.3 * pi, 0.3 * pi)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 0, 0] = True
    _mark(state, self.shot, 1 if self.fired else 4)
    for tank in self.tanks:
      _mark(state, tank, 2)
    for bullet in self.bullets:
      _mark(state, bullet, 3)
    values = (self.steps, self.action, self.multiplier, round((self.shot.vy + 16) * 16),
              round(self.timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _inside(obj):
  return 0 <= obj.x < 150 and 0 <= obj.y < 100


def _mark(state, obj, channel):
  if _inside(obj):
    state[int(obj.y // 10), int(obj.x // 15), channel] = True
