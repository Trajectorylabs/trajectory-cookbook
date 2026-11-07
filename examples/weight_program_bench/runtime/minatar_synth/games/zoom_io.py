from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Arrow:
  x: float
  y: float
  vx: float
  vy: float
  enemy: bool


class ZoomIoEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle, self.speed, self.zoom = -pi / 2, 0., 1.
    self.multiplier = 1.
    self.arrows = []
    self.timer = self.count = 0
    self.spawn_angles = self.random.uniform(0, 2 * pi, 2)
    self.action = self.steps = 0
    self.terminal = False

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
    self.timer -= 1
    if self.timer < 0:
      self._spawn()
    if self.action == 5:
      self.zoom = min(9, self.zoom + 0.05)
      self.multiplier += self.zoom * 0.1
    else:
      self.zoom += (1 - self.zoom) * 0.03
      self.angle = (self.angle + 0.1 / self.zoom) % (2 * pi)
    if self.zoom < 2:
      self.multiplier += (0.5 - self.multiplier) * 0.02
    self.speed += ((self.zoom - 1) * 0.1 - self.speed) * 0.05
    dx, dy = self.speed * cos(self.angle), self.speed * sin(self.angle)
    reward = 0
    remaining = []
    for arrow in self.arrows:
      arrow.x += arrow.vx - dx
      arrow.y += arrow.vy - dy
      if hypot(arrow.x, arrow.y) < 4.5:
        if arrow.enemy:
          self.terminal = True
        else:
          reward += ceil(self.multiplier)
          continue
      if hypot(arrow.x, arrow.y) <= 70:
        remaining.append(arrow)
    self.arrows = remaining
    return reward

  def _spawn(self):
    side = self.count % 2
    if self.random.random() < 0.1:
      angle = self.random.uniform(0, 2 * pi)
      self.spawn_angles[side] = self.angle if side and self.zoom > 5 else angle
    angle = self.spawn_angles[side]
    x, y = 70 * cos(angle), 70 * sin(angle)
    angle = self.random.uniform(0, 2 * pi)
    radius = self.random.uniform(20, 40)
    heading = atan2(radius * sin(angle) - y, radius * cos(angle) - x)
    speed = self.random.uniform(0.1, 0.11)
    self.arrows.append(Arrow(x, y, speed * cos(heading), speed * sin(heading), bool(side)))
    self.timer = 20
    self.count += 1

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    _mark(state, cos(self.angle) * 2 * self.zoom, sin(self.angle) * 2 * self.zoom, 0)
    for arrow in self.arrows:
      distance = hypot(arrow.x, arrow.y)
      x, y = arrow.x * self.zoom, arrow.y * self.zoom
      if sqrt(self.zoom) > 0.5 + distance * 0.03 and -50 <= x < 49 and -50 <= y < 49:
        _mark(state, x, y, 2 if arrow.enemy else 1)
      elif distance:
        for radius in (10, 20, 30):
          _mark(state, radius * arrow.x / distance, radius * arrow.y / distance,
                4 if arrow.enemy else 3)
    values = (self.steps, self.action, round(self.zoom * 100), round(self.speed * 100),
              ceil(self.multiplier), round((self.angle % (2 * pi)) * 128), self.timer)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    for row, value in enumerate(values):
      state[8, row, 5] = bool(value & (1 << 10))
    return state


def _mark(state, x, y, channel):
  state[int(np.clip((y + 50) // 10, 0, 9)), int(np.clip((x + 50) // 10, 0, 9)), channel] = True
