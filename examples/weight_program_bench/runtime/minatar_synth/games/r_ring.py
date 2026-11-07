from collections import deque
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass
class Ring:
  x: float
  y: float
  radius: float = 1
  hit: bool = False

  def arcs(self):
    r = self.radius
    return (
      self._arc(self.x - r * 0.32, self.y, r * 0.2, 5 * pi / 6, 7 * pi / 6),
      self._arc(self.x + r * 0.32, self.y, r * 0.2, -pi / 6, pi / 6),
      self._arc(self.x, self.y + r * 0.3, r * 0.6, -3 * pi / 4, -pi / 4),
    )

  @staticmethod
  def _arc(x, y, radius, start, end):
    angles = np.linspace(start, end, max(2, int(radius * (end - start)) + 2))
    return x + radius * np.cos(angles), y + radius * np.sin(angles)


class RRingEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 50.0
    self.ship_x = self.ship_y = 20.0
    self.target_x = self.target_y = 20.0
    self.angle = 0.0
    self.move_ticks = self.stop_ticks = self.fire_ticks = 0
    self.power_ticks = 60
    self.options = self.action = self.steps = 0
    self.max_options = self.multiplier = 1
    self.history = deque(maxlen=99)
    self.followers = []
    self.rings = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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
    self.x = min(99, max(0, self.x + 3 * ((self.action == 3) - (self.action == 1))))
    self.stop_ticks -= 1
    if self.stop_ticks < 0:
      self.move_ticks = self.random.uniform(120, 180)
      self.stop_ticks = self.move_ticks + self.random.uniform(40, 60)
    self.move_ticks -= 1
    if self.move_ticks > 0:
      if hypot(self.target_x - self.ship_x, self.target_y - self.ship_y) < 5:
        self.target_x = self.random.uniform(10, 90)
        self.target_y = self.random.uniform(3, 30)
      target = atan2(self.target_y - self.ship_y, self.target_x - self.ship_x)
      turn = (target - self.angle + pi) % (2 * pi) - pi
      self.angle += min(0.1, max(-0.1, turn))
      self.ship_x = min(99, max(0, self.ship_x + 0.5 * cos(self.angle)))
      self.ship_y = min(50, max(0, self.ship_y + 0.5 * sin(self.angle)))
      self.history.appendleft((self.ship_x, self.ship_y))
    self.power_ticks -= 1
    if self.power_ticks < 0:
      self.options += 1
      if self.options > self.max_options:
        self.options = 0
        self.max_options = min(4, self.max_options + 1)
      self.power_ticks = 300 / self.options if self.options else 60
    self.followers = [self.history[i * 24] for i in range(1, self.options + 1)
                      if i * 24 < len(self.history)]
    self.fire_ticks -= 1
    if self.fire_ticks < 0:
      self.rings.append(Ring(self.ship_x, self.ship_y))
      self.rings.extend(Ring(x, y) for x, y in self.followers)
      self.fire_ticks = 60
    reward = 0
    remaining = []
    for ring in self.rings:
      ring.y += 1
      ring.radius += 0.5
      left, right, center = ring.arcs()
      self.terminal |= self._touches(left) or self._touches(right)
      if not ring.hit and self._touches(center):
        reward += self.multiplier
        self.multiplier += 1
        ring.hit = True
      if ring.y <= 110:
        remaining.append(ring)
      elif not ring.hit:
        self.multiplier = max(1, self.multiplier - 1)
    self.rings = remaining
    return reward

  def _touches(self, arc):
    x, y = arc
    return bool(np.any((abs(x - self.x) < 4) & (abs(y - 95) < 4)))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x // 10), 0] = True
    for x, y in [(self.ship_x, self.ship_y), *self.followers]:
      state[int(y // 10), int(x // 10), 1] = True
    for ring in self.rings:
      for channel, (xs, ys) in zip((2, 2, 4 if ring.hit else 3), ring.arcs()):
        visible = (0 <= xs) & (xs < 100) & (0 <= ys) & (ys < 100)
        state[(ys[visible] // 10).astype(int), (xs[visible] // 10).astype(int), channel] = True
    for row, value in enumerate((self.steps, self.multiplier, self.fire_ticks, self.options)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
