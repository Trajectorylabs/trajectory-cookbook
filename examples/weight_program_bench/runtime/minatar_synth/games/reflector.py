from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass
class Tank:
  x: float
  speed: float
  turn: float
  interval: float
  timer: float
  shot_speed: float
  angle: float = -pi / 2


@dataclass
class Bullet:
  x: float
  y: float
  vx: float
  vy: float


@dataclass
class Blast:
  x: float
  y: float
  radius: float
  duration: float
  age: int = 0

  def reach(self):
    return self.radius * sin(pi * self.age / self.duration)


class ReflectorEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.direction = 40.0, 1
    self.angle = self.power = 0.0
    self.timer = self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.tanks = []
    self.bullets = []
    self.blasts = []

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
    if tap or (self.x < 3 and self.direction < 0) or (self.x > 97 and self.direction > 0):
      self.direction *= -1
    if self.held:
      self.power += (1 - self.power) * 0.05
    else:
      self.power *= 0.9
      self.angle = min(pi / 4, max(-pi / 4, self.angle - self.direction * 0.07))
    self.x += self.direction * (0.25 if self.held else 0.5)
    if not self.tanks:
      self.timer = 0
    self.timer -= 1
    if self.timer < 0:
      direction = self.random.choice((-1, 1))
      interval = self.random.uniform(200, 300)
      agility = self.random.uniform(1, 5)
      self.tanks.append(Tank(-3 if direction > 0 else 103, direction * 0.1 / agility ** 0.5,
                             agility * 0.002, interval, interval, self.random.uniform(1, 1.5)))
      self.timer = self.random.uniform(60, 80)
    if not self.blasts:
      self.multiplier = 1
    for blast in self.blasts:
      blast.age += 1
    self.blasts = [blast for blast in self.blasts if blast.age <= blast.duration]
    reward = 0
    tanks = []
    for tank in self.tanks:
      tank.x += tank.speed
      aim = atan2(-78, self.x - tank.x)
      tank.angle += min(tank.turn, max(-tank.turn, aim - tank.angle))
      if any(hypot(tank.x - blast.x, 87 - blast.y) <= blast.reach() + 3 for blast in self.blasts):
        reward += self.multiplier
        self.multiplier += 1
        continue
      tank.timer -= 1
      if tank.timer < 0:
        self.bullets.append(Bullet(tank.x, 87, cos(tank.angle) * tank.shot_speed,
                                   sin(tank.angle) * tank.shot_speed))
        tank.timer = tank.interval
      if -3 <= tank.x <= 103:
        tanks.append(tank)
    self.tanks = tanks
    bullets = []
    for bullet in self.bullets:
      bullet.x += bullet.vx
      bullet.y += bullet.vy
      if bullet.vy < 0:
        if self._shield_hit(bullet):
          angle = 2 * self.angle - atan2(bullet.vy, bullet.vx)
          speed = hypot(bullet.vx, bullet.vy) * (1 + 4 * self.power)
          bullet.vx, bullet.vy = speed * cos(angle), speed * sin(angle)
          bullet.x += bullet.vx * 2
          bullet.y = max(20, bullet.y + bullet.vy * 2)
        elif abs(bullet.x - self.x) < 5 and abs(bullet.y - 9) < 5:
          self.terminal = True
      if bullet.vy > 0 and (bullet.y >= 88 or any(
        abs(bullet.x - tank.x) < 5 and abs(bullet.y - 87) < 5 for tank in self.tanks
      )):
        speed = hypot(bullet.vx, bullet.vy)
        self.blasts.append(Blast(bullet.x, bullet.y, speed ** 2, speed * 9))
      elif -3 <= bullet.x <= 103 and -3 <= bullet.y <= 103:
        bullets.append(bullet)
    self.bullets = bullets
    return reward

  def _shield_hit(self, bullet):
    dx = bullet.x - (self.x - 6 * sin(self.angle))
    dy = bullet.y - (9 + 6 * cos(self.angle))
    parallel = dx * cos(self.angle) + dy * sin(self.angle)
    normal = -dx * sin(self.angle) + dy * cos(self.angle)
    return abs(parallel) < (9 - self.power * 9) / 2 + 2 and abs(normal) < (3 + self.power * 3) / 2 + 2

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[0, int(self.x // 10), 0] = True
    for tank in self.tanks:
      if 0 <= tank.x < 100:
        state[8, int(tank.x // 10), 1] = True
    for bullet in self.bullets:
      if 0 <= bullet.x < 100 and 0 <= bullet.y < 100:
        state[int(bullet.y // 10), int(bullet.x // 10), 2 + int(bullet.vy >= 0)] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for blast in self.blasts:
      state[:, :, 4] |= np.hypot(cols - blast.x, rows - blast.y) <= blast.reach() + 5
    gauges = (self.steps, self.multiplier, round(self.power * 100),
              round((self.angle + pi / 4) * 100), self.direction > 0, self.held)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
