from dataclasses import dataclass
from math import atan2, cos, floor, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Star:
  x: float
  y: float
  radius: float


class SwingbyEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = self.vx = self.vy = self.anchor_x = self.anchor_y = 0.
    self.screen_x = self.screen_y = 50.
    self.damage = self.steps = self.score = 0
    self.held = self.terminal = False
    self.stars = [Star(0, 0, 20)]
    for _ in range(20):
      self._spawn(0, 0, 150, -pi, pi)
    self.stars.pop(0)
    self.display = []

  def _spawn(self, x, y, reach, start, stop):
    for _ in range(99):
      angle = self.random.uniform(start, stop)
      distance = self.random.uniform(0, reach)
      px, py = x + distance * cos(angle), y + distance * sin(angle)
      radius = self.random.uniform(5, 15)
      if all(hypot(px - star.x, py - star.y) >= radius + star.radius + 30 for star in self.stars):
        self.stars.append(Star(px, py, radius))
        break

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    previous = self.score
    for _ in range(4):
      self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(self.score - previous), self.terminal

  def _advance(self):
    remaining = []
    self.display = []
    collision = False
    for star in self.stars:
      dx, dy = star.x - self.x, star.y - self.y
      distance = hypot(dx, dy)
      force = star.radius * 0.01 / min(99, max(2, distance - star.radius))
      angle = atan2(dy, dx)
      self.vx += force * cos(angle)
      self.vy += force * sin(angle)
      if distance < 99 + star.radius:
        self.display.append((dx, dy, star.radius))
        collision |= abs(distance - (star.radius - 2)) < 4.5
      if distance <= 150:
        remaining.append(star)
    self.stars = remaining
    dx, dy = self.x - self.anchor_x, self.y - self.anchor_y
    if hypot(dx, dy) > 50:
      angle = atan2(dy, dx)
      self.anchor_x, self.anchor_y = self.x, self.y
      for _ in range(5):
        self._spawn(self.x + 100 * cos(angle), self.y + 100 * sin(angle),
                    70, angle - pi / 2, angle + pi / 2)
    self.vx *= 0.99
    self.vy *= 0.99
    angle = atan2(self.vy, self.vx)
    lead = min(30, max(1, sqrt(hypot(self.vx, self.vy)) * 19))
    self.screen_x, self.screen_y = 50 - lead * cos(angle), 50 - lead * sin(angle)
    if self.held:
      self.vx -= 0.05 * sin(angle)
      self.vy += 0.05 * cos(angle)
    angle = atan2(self.vy, self.vx)
    self.vx += 0.01 * cos(angle)
    self.vy += 0.01 * sin(angle)
    self.x += self.vx
    self.y += self.vy
    if collision:
      self.vx *= 0.99
      self.vy *= 0.99
      self.damage += 4
    else:
      self.damage = max(0, self.damage - 1)
    self.terminal = self.damage > 99
    self.score = floor(hypot(self.x, self.y))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = self.screen_x, self.screen_y
    state[int(y // 10), int(x // 10), 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for dx, dy, radius in self.display:
      state[:, :, 1] |= abs(np.hypot(cols - x - dx, rows - y - dy) - (radius - 2)) < 8
    origin = np.clip(np.array([x - self.x, y - self.y]) // 10, 0, 9).astype(int)
    state[origin[1], origin[0], 2] = True
    angle = atan2(self.vy, self.vx)
    state[int((y + 10 * sin(angle)) // 10), int((x + 10 * cos(angle)) // 10), 3] = True
    state[int(y // 10), int(x // 10), 4] = self.damage > 0
    values = (self.steps, self.damage, self.held, min(2047, self.score),
              round((self.vx + 10) * 50), round((self.vy + 10) * 50))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit % 10, 5] = bool(value & (1 << bit))
    state[6, 0, 5] = self.score >= 1024
    return state
