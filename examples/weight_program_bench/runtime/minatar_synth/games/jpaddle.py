from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Ball:
  x: float
  y: float
  vx: float
  vy: float


class JpaddleEnvironment:
  score_cap = 600

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 90.0
    self.jumps = self.steps = self.frames = 0
    self.direction = -1
    self.pressed = self.terminal = False
    self.balls: list[Ball] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, tap: bool) -> float:
    if tap and self.jumps < 99:
      self.jumps += 1
      self.y -= 12 // self.jumps
      self.direction *= -1
    else:
      self.y += 0.1
    if self.pressed and self.jumps:
      self.x += 0.5 * self.direction
    self.x = min(97, max(3, self.x))
    self.y = min(110, max(30, self.y))
    reward = 0.0
    remaining = []
    for ball in self.balls:
      ball.x += ball.vx
      ball.y += ball.vy
      if ball.x < 0 or ball.x > 97:
        ball.vx *= -1
      if ball.vy > 0 and abs(ball.x - self.x) < 7.5 and abs(ball.y - self.y) < 6.5:
        ball.y = self.y - 3
        ball.vy *= -1
        reward += 99 - ball.y
        self.jumps = max(0, self.jumps - 9)
      if ball.y <= 100:
        remaining.append(ball)
    self.balls = remaining
    if self.frames % 60 == 0:
      x = self.random.uniform(10, 90)
      vx = self.random.uniform(-1, 1)
      length = math.hypot(vx, 1)
      self.balls.append(Ball(x, 0, vx / length, 1 / length))
    self.frames += 1
    self.terminal = self.y > 99
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = min(9, int(self.y // 10)), int(self.x // 10)
    state[row, col, 0] = True
    for ball in self.balls:
      if 0 <= ball.y < 100 and 0 <= ball.x < 100:
        state[int(ball.y // 10), int(ball.x // 10), 1 if ball.vy > 0 else 2] = True
        x, y = ball.x + 8 * ball.vx, ball.y + 8 * ball.vy
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), 3] = True
    state[9, :, 4] = True
    for row, value in enumerate((self.steps, self.jumps, round(self.x * 4), round(self.y * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
