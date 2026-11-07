from dataclasses import dataclass
from math import hypot

import numpy as np


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Tank:
  x: float
  y: float
  heading: int
  shot: float = 0
  speed: float = 0
  burst: int = 0
  turn: float = 0
  reflecting: bool = False


@dataclass
class Bullet:
  x: float
  y: float
  heading: int
  friendly: bool
  life: int = 20


class ZoneBEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.action = self.steps = self.level = 0
    self.terminal = False
    self._start_level()

  def _start_level(self):
    self.enemies = [
      Tank(*self.random.uniform(0, 99, 2), int(self.random.integers(4)),
           self.random.uniform(200, 300))
      for _ in range(9 + self.level * 7)
    ]
    self.walls = []
    for _ in range(19):
      x, y = self.random.uniform(9, 89, 2)
      width = self.random.uniform(2.5, 7.5)
      self.walls.append((x, y, width, 1) if self.random.integers(2) else (x, y, 1, width))
    self.player = Tank(50, 80, 3)
    self.bullets = []
    self.center = np.array([50., 50.])
    self.radius = 120.
    self.next_center = self.center.copy()
    self.next_radius = 60.
    self.timer = 99
    self.multiplier = 1
    self.clear = 60
    self.level += 1

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    release = previous == 5 and self.action != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, release):
    self._zone()
    for bullet in self.bullets:
      dx, dy = _DIRECTIONS[bullet.heading]
      bullet.x += dx
      bullet.y += dy
      bullet.life -= 1
    self.bullets = [b for b in self.bullets if b.life >= 0 and _inside(b) and not self._wall(b, 1)]
    reward = 0
    remaining = []
    for enemy in self.enemies:
      dx, dy = _DIRECTIONS[enemy.heading]
      enemy.x += dx * enemy.speed
      enemy.y += dy * enemy.speed
      if any(b.friendly and _distance(enemy, b) < 3.5 for b in self.bullets):
        reward += self.multiplier
        self.multiplier += 1
        continue
      self._reflect(enemy)
      if enemy.shot > 7:
        enemy.turn -= 1
      if enemy.turn < 0:
        enemy.heading = (enemy.heading + 1) % 4
        enemy.turn = self.random.uniform(200, 300)
      enemy.shot -= 1
      if enemy.shot < 0:
        enemy.burst -= 1
        if enemy.burst < 0:
          enemy.shot = self.random.uniform(100, 200)
          enemy.burst = int(self.random.integers(3, 7))
        else:
          self._shoot(enemy, False)
          enemy.shot += 7
      enemy.speed += ((0 if enemy.shot < 7 else 0.2) - enemy.speed) * 0.1
      remaining.append(enemy)
    self.enemies = remaining
    p = self.player
    dx, dy = _DIRECTIONS[p.heading]
    p.speed += ((0 if self.action == 5 else 0.2) - p.speed) * 0.1
    if release and p.speed > 0.04:
      p.heading = (p.heading + 1) % 4
    p.shot -= 1
    if self.action == 5 and p.speed < 0.04 and p.shot < 0:
      self._shoot(p, True)
      p.shot = 7
    p.x += dx * p.speed
    p.y += dy * p.speed
    self.terminal = (self.radius < 1 or hypot(p.x - self.center[0], p.y - self.center[1]) > self.radius * 1.05
                     or any(not b.friendly and _distance(p, b) < 3.5 for b in self.bullets))
    self._reflect(p)
    if not self.enemies and not self.terminal:
      self.clear -= 1
      if self.clear == 0:
        self._start_level()
    return reward

  def _zone(self):
    self.timer -= 1
    if self.timer == 9:
      for _ in range(99):
        candidate = self.random.uniform(10, 90, 2)
        if np.linalg.norm(candidate - self.next_center) < 10:
          break
      self.zone_target = candidate
    if self.timer < 0:
      self.timer = 600
    if self.timer < 9:
      self.next_center += (self.zone_target - self.next_center) / (self.timer + 1)
      if self.timer == 0:
        self.next_radius -= 10
    if self.next_radius < 60 and self.timer > 9:
      self.center += (self.next_center - self.center) / self.timer
      self.radius += (self.next_radius - self.radius) / self.timer
    self.walls = [(x, y, w, h) for x, y, w, h in self.walls
                  if abs(hypot(x - self.center[0], y - self.center[1]) - self.radius) > hypot(w, h)]

  def _wall(self, obj, margin):
    return any(abs(x - obj.x) < w + margin and abs(y - obj.y) < h + margin
               for x, y, w, h in self.walls)

  def _reflect(self, tank):
    blocked = not _inside(tank) or self._wall(tank, 2)
    if blocked and not tank.reflecting:
      tank.heading = (tank.heading + 2) % 4
    tank.reflecting = blocked
    dx, dy = self.center - (tank.x, tank.y)
    if hypot(dx, dy) >= self.radius:
      tank.heading = (0 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 3)
    tank.x = min(99, max(0, tank.x))
    tank.y = min(99, max(0, tank.y))

  def _shoot(self, tank, friendly):
    dx, dy = _DIRECTIONS[tank.heading]
    self.bullets.append(Bullet(tank.x + 5 * dx, tank.y + 5 * dy, tank.heading, friendly))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    _mark(state, self.player, 0)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for x, y, w, h in self.walls:
      state[:, :, 1] |= (abs(cols - x) < w + 5) & (abs(rows - y) < h + 5)
    state[:, :, 1] |= np.hypot(cols - self.center[0], rows - self.center[1]) >= self.radius
    for enemy in self.enemies:
      if _distance(enemy, self.player) < 30:
        _mark(state, enemy, 2)
    for bullet in self.bullets:
      if bullet.friendly or _distance(bullet, self.player) < 30:
        _mark(state, bullet, 3 if bullet.friendly else 4)
    values = (self.steps, self.action, self.player.heading, round(self.player.speed * 100),
              self.multiplier, self.level, self.timer, round(self.radius * 4))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _inside(obj):
  return 0 <= obj.x < 99 and 0 <= obj.y < 99


def _distance(first, second):
  return hypot(first.x - second.x, first.y - second.y)


def _mark(state, obj, channel):
  if 0 <= obj.x < 100 and 0 <= obj.y < 100:
    state[int(obj.y // 10), int(obj.x // 10), channel] = True
