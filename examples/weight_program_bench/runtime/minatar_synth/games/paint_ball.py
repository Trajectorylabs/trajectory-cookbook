from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Ball:
  x: float
  y: float
  vx: float
  vy: float
  color: int
  paint: int = 0


class PaintBallEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.grid = np.zeros((15, 12), dtype=np.uint8)
    self.offset = 0.0
    self.x, self.y, self.angle = 9.0, 70.0, 0.0
    self.direction = -1
    self.timer = self.steps = 0
    self.multiplier = 1
    self.ready = True
    self.pressed = self.terminal = False
    self.balls: list[Ball] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    self.offset += 0.03
    if self.offset > 4:
      self.offset -= 7
      self.grid[1:] = self.grid[:-1].copy()
      self.grid[0] = 0
    if self.ready:
      self.y = min(95, self.y + 0.03)
      self.angle += self.direction * 0.02
      if self.angle * self.direction > math.pi / 4:
        self.direction *= -1
      if tap:
        angle = self.angle if self.x < 50 else math.pi - self.angle
        self.balls.append(Ball(self.x, self.y, 2 * math.cos(angle), 2 * math.sin(angle), 1))
        self.ready = False
        self.multiplier = 1
    self.timer -= 1
    if self.timer < 0:
      angle = self.random.uniform(math.pi / 8, math.pi * 7 / 8)
      self.balls.append(Ball(self.random.uniform(20, 80), -3,
                             0.1 * math.cos(angle), 0.1 * math.sin(angle), 2))
      self.timer = 150
    reward = 0
    remaining = []
    for ball in self.balls:
      col = math.floor((ball.x - 8.5) / 7)
      row = math.floor((ball.y + 2 - self.offset) / 7)
      inside = 0 <= row < 15 and 0 <= col < 12
      tile = self.grid[row, col] if inside else 0
      slow = ball.color == 2 and tile == 1
      ball.paint = ball.paint + 1 if slow else 999
      speed = 0.1 if slow else 1
      ball.x += ball.vx * speed
      ball.y += ball.vy * speed + 0.03
      if (ball.x <= 9 and ball.vx < 0) or (ball.x >= 90 and ball.vx > 0):
        if ball.color == 1:
          self.x, self.y, self.angle = ball.x, ball.y, 0
          self.direction, self.multiplier, self.ready = -1, 1, True
          continue
        ball.vx *= -1
      self.terminal |= ball.color == 2 and ball.y > 99
      if (ball.y < 3 and ball.vy < 0) or (ball.y > 99 and ball.vy > 0):
        ball.vy *= -1
      if inside and (ball.color == 1 or ball.paint > 99):
        if ball.color == 1 and tile != 1:
          self.multiplier += tile == 2
          reward += self.multiplier
        self.grid[row, col] = ball.color
        ball.paint = 0
      remaining.append(ball)
    player = [b for b in remaining if b.color == 1]
    targets = [(b.x, b.y) for b in player]
    if self.ready:
      targets.append((self.x, self.y))
    self.balls = []
    for ball in remaining:
      if ball.color == 2 and any(abs(ball.x - x) < 5 and abs(ball.y - y) < 5 for x, y in targets):
        self.multiplier += 1
        reward += self.multiplier
      else:
        self.balls.append(ball)
    if not self.balls:
      self.timer = 0
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    pairs = self.grid.reshape(9, 10, 2)
    packed = pairs[:, :, 0] + 3 * pairs[:, :, 1]
    for bit in range(4):
      state[:9, :, bit] = (packed & (1 << bit)) != 0
    for ball in self.balls:
      if 0 <= ball.y < 100 and 0 <= ball.x < 100:
        state[int(ball.y // 10), int(ball.x // 10), 3 + ball.color] = True
    if self.ready:
      state[min(9, max(0, int(self.y // 10))), int(self.x // 10), 4] = True
    values = (self.steps, min(511, self.multiplier), round((self.angle + 1) * 100), round((self.offset + 3) * 30))
    for channel, value in enumerate(values):
      for bit in range(9):
        state[9, bit, channel] = bool(value & (1 << bit))
    state[9, 9, :3] = self.pressed, self.ready, self.direction > 0
    return state
