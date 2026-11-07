from dataclasses import dataclass
from math import cos, hypot, pi, sin

import numpy as np


@dataclass
class Bar:
  x: float
  y: float
  angle: float
  length: float

  def coordinates(self, x, y):
    dx, dy = x - self.x, y - self.y
    return dx * cos(self.angle) + dy * sin(self.angle), dy * cos(self.angle) - dx * sin(self.angle)


@dataclass
class Ball:
  x: float
  y: float
  vx: float
  vy: float = 0
  age: int = 0


class TiltedEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.bar = Bar(50, 0, 0, 90)
    self.bars = [self.bar]
    self.balls = []
    self.bar_distance, self.bar_count, self.ball_timer = 10, 0, 60
    self.x, self.y, self.vx, self.vy = 50., -6., 1., 0.
    self.jumps = self.action = self.steps = 0
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
    pressed = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0.
    for frame in range(4):
      reward += self._advance(pressed and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, pressed):
    scroll = 0.02 + max(0, 60 - self.y) * 0.1
    if self.bar is not None:
      bar = self.bar
      distance = hypot(self.x - bar.x, self.y + 5 - bar.y)
      distance = (-distance if self.x < bar.x else distance) + self.vx / 2
      self.x = bar.x + cos(bar.angle) * distance
      self.y = bar.y + sin(bar.angle) * distance - 5
      if abs(distance) > bar.length / 2 + 6:
        self.bar = None
        self.jumps = 0
      if pressed:
        self.bar = None
        self.vy, self.jumps = -2, 1
        self.y -= 2
    else:
      if pressed:
        if self.jumps:
          self.vx *= -1
        self.jumps = max(1, self.jumps)
        self.vy = -2 / self.jumps ** 2
        self.jumps += 1
      self.vy += 0.05 if self.action == 5 else 0.1
      self.x += self.vx / 2
      self.y += scroll + self.vy / 2
    if (self.x < 8 and self.vx < 0) or (self.x > 92 and self.vx > 0):
      self.vx *= -1
    if self.y > 99:
      self.terminal = True
      return scroll
    self._advance_balls(scroll)
    for bar in self.bars:
      bar.y += scroll
      along, normal = bar.coordinates(self.x, self.y + 3)
      if self.bar is None and abs(along) <= bar.length / 2 + 2 and abs(normal) <= 4:
        if self.vy <= 0:
          self.vy *= -0.5
          self.y += 3 * self.vy
        else:
          self.bar = bar
      for ball in self.balls:
        along, normal = bar.coordinates(ball.x, ball.y)
        if abs(along) <= bar.length / 2 + 3 and abs(normal) <= 4:
          tangent_x, tangent_y = cos(bar.angle), sin(bar.angle)
          projection = ball.vx * tangent_x + ball.vy * tangent_y
          ball.vx = 2 * projection * tangent_x - ball.vx
          ball.vy = 2 * projection * tangent_y - ball.vy
          ball.x += ball.vx
          ball.y += ball.vy
          ball.age += 30
          break
    self.bars = [bar for bar in self.bars if bar.y <= 120]
    self.bar_distance -= scroll
    while self.bar_distance < 0:
      self._spawn_bar()
    return scroll

  def _advance_balls(self, scroll):
    self.ball_timer -= 1
    if self.ball_timer < 0:
      speed = self.random.choice((-1, 1)) * self.random.uniform(0.5, 1)
      self.balls.append(Ball(self.random.uniform(20, 80), -40, speed))
      self.ball_timer += self.random.uniform(60, 80)
    for ball in self.balls:
      ball.vx *= 0.995
      ball.vy = (ball.vy + 0.05) * 0.995
      ball.x += ball.vx / 2
      ball.y += scroll + ball.vy / 2
      if (ball.x < 8 and ball.vx < 0) or (ball.x > 92 and ball.vx > 0):
        ball.vx *= -1
      ball.age += 1
      if hypot(ball.x - self.x, ball.y - self.y) <= 5:
        if self.bar is not None:
          self.bar.y = 999
          self.bar = None
          self.vy += 1
        else:
          self.vy += 4
        ball.age = 999
    for index, ball in enumerate(self.balls):
      for other in self.balls[index + 1:]:
        if ball.age <= 500 and other.age <= 500 and hypot(ball.x - other.x, ball.y - other.y) <= 5:
          ball.age = other.age = 999
    self.balls = [ball for ball in self.balls if ball.age <= 500 and ball.y <= 103]

  def _spawn_bar(self):
    side = self.bar_count % 2
    length = self.random.uniform(15, 22)
    x = (50 + self.random.choice((-1, 1)) * self.random.uniform(25, 50 - length / 2)
         if side else self.random.uniform(35, 65))
    angle = 0. if self.random.random() < (0.2 if side else 0.7) else (
      self.random.choice((-1, 1)) * self.random.uniform(0, pi / 5)
    )
    y = -9 - self.bar_distance
    self.bars.append(Bar(x, y, angle, length))
    if side:
      self.bars.append(Bar(100 - x, y, -angle, length))
    self.bar_distance += self.random.uniform(15, 20)
    self.bar_count += 1

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(np.clip(self.x // 10, 0, 9)), 0] = True
    for bar in self.bars:
      for offset in np.linspace(-bar.length / 2, bar.length / 2, int(bar.length / 3) + 2):
        x = bar.x + offset * cos(bar.angle)
        y = bar.y + offset * sin(bar.angle)
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), 2 if bar is self.bar else 1] = True
    for ball in self.balls:
      if 0 <= ball.y < 100:
        state[int(ball.y // 10), int(ball.x // 10), 3] = True
    state[:, (0, 9), 4] = True
    gauges = (self.steps, self.action, self.vx > 0, self.jumps, round((self.vy + 8) * 16))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
