from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass
class Grenade:
  x: float
  y: float
  tx: float
  ty: float
  age: int = 0


@dataclass
class Smoke:
  x: float
  y: float
  radius: float = 0
  growing: bool = True


@dataclass
class Enemy:
  x: float
  y: float
  angle: float


class SmokeGEnvironment:
  score_cap = 20
  moves = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50, 45
    self.angle = -pi / 2
    self.steps = self.action = self.side_count = 0
    self.side = 1
    self.distance = 0.
    self.terminal = False
    self.grenades, self.smoke, self.enemies = [], [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    tap = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    dx, dy = self.moves[self.action]
    self.x = min(99, max(0, self.x + 3 * dx))
    self.y = min(99, max(0, self.y + 3 * dy))
    if tap and len(self.smoke) < 9:
      self.grenades.append(Grenade(50, 90, self.x, self.y))
    for grenade in self.grenades:
      grenade.age += 1
      grenade.x += (grenade.tx - grenade.x) * 0.1
      grenade.y += (grenade.ty - grenade.y) * 0.1 + cos(grenade.age * pi * 4 / 30) + 0.1
      if grenade.age > 30:
        self.smoke.append(Smoke(grenade.tx, grenade.ty))
    self.grenades = [grenade for grenade in self.grenades if grenade.age <= 30]
    for smoke in self.smoke:
      smoke.y += 0.1
      if smoke.growing:
        smoke.radius += (10 - smoke.radius) * 0.2
        if smoke.radius > 9:
          smoke.radius, smoke.growing = 10, False
      else:
        smoke.radius *= 0.995
    self.smoke = [smoke for smoke in self.smoke if smoke.growing or smoke.radius >= 2]
    if not self.enemies:
      self.distance = 0
    self.distance -= 0.1
    if self.distance < 0:
      self.side_count -= 1
      if self.side_count < 0:
        self.side *= -1
        self.side_count = self.random.uniform(1, 5)
        self.distance += 7
      x = 50 + self.random.uniform(0, 40) * self.side
      self.enemies.append(Enemy(x, -3, atan2(-93, x - 50) + self.random.uniform(-0.2, 0.2)))
      self.distance += self.random.uniform(5, 9)
    visible = [enemy for enemy in self.enemies
               if hypot(enemy.x - 50, enemy.y - 90) < 99 and not self._blocked(enemy)]
    if visible:
      target = min(visible, key=lambda enemy: hypot(enemy.x - 50, enemy.y - 90))
      self.angle = self._turn(self.angle, atan2(target.y - 90, target.x - 50), 0.012)
    reward = 0
    remaining = []
    for enemy in self.enemies:
      enemy.y += 0.1
      if not self._blocked(enemy):
        enemy.angle = self._turn(enemy.angle, atan2(90 - enemy.y, 50 - enemy.x), 0.008)
        self.terminal |= self._ray_hit(enemy.x, enemy.y, enemy.angle, 50, 90, 50)
      if self._ray_hit(50, 90, self.angle, enemy.x, enemy.y, 55):
        reward += 1
      else:
        self.terminal |= enemy.y > 99
        remaining.append(enemy)
    self.enemies = remaining
    return reward

  def _blocked(self, enemy):
    dx, dy = enemy.x - 50, enemy.y - 90
    length = dx * dx + dy * dy
    for smoke in self.smoke:
      t = min(1, max(0, ((smoke.x - 50) * dx + (smoke.y - 90) * dy) / max(length, 1e-9)))
      nearest = hypot(smoke.x - 50 - t * dx, smoke.y - 90 - t * dy)
      farthest = max(hypot(smoke.x - 50, smoke.y - 90), hypot(smoke.x - enemy.x, smoke.y - enemy.y))
      if nearest <= smoke.radius + 1 and farthest >= smoke.radius - 1:
        return True
    return False

  @staticmethod
  def _turn(angle, target, speed):
    difference = (target - angle + pi) % (2 * pi) - pi
    return angle + min(speed, max(-speed, difference))

  @staticmethod
  def _ray_hit(x, y, angle, tx, ty, reach):
    dx, dy = tx - x, ty - y
    along = dx * cos(angle) + dy * sin(angle)
    across = dy * cos(angle) - dx * sin(angle)
    return -4 < along < reach + 4 and abs(across) < 4

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 5, 0] = True
    state[self.y // 10, self.x // 10, 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.y < 100:
        state[int(enemy.y // 10), int(enemy.x // 10), 2] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for smoke in self.smoke:
      state[:, :, 3] |= abs(np.hypot(cols - smoke.x, rows - smoke.y) - smoke.radius) < 6
    for grenade in self.grenades:
      if 0 <= grenade.y < 100:
        state[int(grenade.y // 10), int(grenade.x // 10), 4] = True
    values = (self.steps, self.action, round((self.angle % (2 * pi)) * 100))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
