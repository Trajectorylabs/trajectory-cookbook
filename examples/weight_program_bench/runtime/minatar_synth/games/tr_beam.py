from dataclasses import dataclass
from math import cos, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Ball:
  x: float
  y: float
  radius: float
  red: bool = False
  vx: float = 0.
  vy: float = 0.
  beamed: bool = False


class TrBeamEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.phase = -pi * 0.3
    self.x = 50 + sin(self.phase) * 40
    self.length = 0.
    self.balls = []
    self.spawn_timer = 0.
    self.until_red = 9
    self.multiplier = 1
    self.action = self.steps = self.frames = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    if self.action == 5 and previous != 5:
      self.multiplier = 1
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.phase += 0.03
    self.x = 50 + sin(self.phase) * 40
    self.length = min(82, self.length + 2) if self.action == 5 else self.length * 0.8
    beams = self._beams()
    self.frames += 1
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self.until_red -= 1
      red = self.until_red < 0
      if red:
        self.until_red = int(self.random.integers(9, 12))
      radius = self.random.uniform(6, 12)
      self.balls.append(Ball(self.random.uniform(10, 90), 105 + radius, radius, red))
      self.spawn_timer = self.random.uniform(20, 25)
    reward = 0
    remaining = []
    for ball in self.balls:
      self._move(ball)
      dx, dy = self.x - ball.x, 9 - ball.y
      distance = hypot(dx, dy)
      if abs(distance - ball.radius) <= 4:
        if ball.red:
          self.terminal = True
          break
        reward += self.multiplier
        self.multiplier += 1
        continue
      if any(_crosses_ring(ball, *beam) for beam in beams):
        strength = 1 / (max(distance, 1e-9) * sqrt(ball.radius))
        ball.vx += dx * (strength + 0.01)
        ball.vy += dy * strength
        ball.beamed = True
      elif ball.beamed:
        ball.vy *= 0.1
        ball.vx *= 5
        ball.beamed = False
      if not ball.beamed:
        if ball.red and (ball.x < ball.radius or ball.x > 99 - ball.radius):
          ball.red = False
        elif 10 <= ball.x < 90 and 12 <= ball.y - ball.radius < 17 and ball.vy > 0:
          ball.red = True
      if -20 <= ball.x <= 120:
        remaining.append(ball)
    if not self.terminal:
      self.balls = remaining
    return reward

  def _move(self, ball):
    if ball.y > 99:
      ball.vy -= 1
      drag = 0.5
    elif ball.x < 0 or ball.x > 99:
      drag = 0.1
    else:
      drag = 0.9
      if ball.red:
        ball.vy *= 0.9
    ball.vx *= drag
    ball.vy = ball.vy * drag + 0.1
    ball.x += ball.vx
    ball.y += ball.vy
    for other in self.balls:
      if other is ball:
        continue
      dx, dy = ball.x - other.x, ball.y - other.y
      distance = hypot(dx, dy)
      overlap = ball.radius + other.radius - distance
      if overlap > 0 and distance > 0:
        scale = overlap / (distance * sqrt(ball.radius))
        ball.vx += dx * scale
        ball.vy += dy * scale

  def _beams(self):
    if self.length <= 1:
      return []
    angle = (self.frames % 10) * pi / 40
    offsets = [cos(angle + index * pi / 4) for index in range(4)]
    return [(self.x + 4 * offset, 14, self.x + 9 * offset, 14 + self.length)
            for offset in offsets]

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[0, int(self.x // 10), 0] = True
    for x1, y1, x2, y2 in self._beams():
      for x, y in zip(np.linspace(x1, x2, 33), np.linspace(y1, y2, 33), strict=True):
        state[int(y // 10), int(x // 10), 1] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for ball in self.balls:
      ring = abs(np.hypot(cols - ball.x, rows - ball.y) - ball.radius) <= 6
      state[:, :, 3 if ball.red else 2] |= ring
      if ball.beamed:
        state[:, :, 4] |= ring
    values = (self.steps, self.action, round(self.length * 4), self.multiplier,
              round((self.phase % (2 * pi)) * 64), self.until_red + 1)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _crosses_ring(ball, x1, y1, x2, y2):
  dx, dy = x2 - x1, y2 - y1
  fraction = ((ball.x - x1) * dx + (ball.y - y1) * dy) / (dx * dx + dy * dy)
  fraction = min(1, max(0, fraction))
  near = hypot(ball.x - x1 - fraction * dx, ball.y - y1 - fraction * dy)
  far = max(hypot(ball.x - x1, ball.y - y1), hypot(ball.x - x2, ball.y - y2))
  return near <= ball.radius + 2 and far >= ball.radius - 2
