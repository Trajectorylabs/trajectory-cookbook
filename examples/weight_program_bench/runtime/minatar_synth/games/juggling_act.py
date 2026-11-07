from dataclasses import dataclass

import numpy as np


@dataclass
class Ball:
  x: float
  y: float = 0.0
  vx: float = 0.0
  vy: float = 0.0


class JugglingActEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.vx = 50.0, 0.0
    self.steps = self.spawn_timer = 0
    self.pressed = self.terminal = False
    self.balls: list[Ball] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _nearest(self) -> Ball | None:
    return min((ball for ball in self.balls if ball.vy >= 0),
               key=lambda ball: abs(90 - ball.y), default=None)

  def _advance(self) -> int:
    if not self.pressed:
      nearest = self._nearest()
      if nearest is not None:
        self.vx += 0.36 if nearest.x > self.x else -0.36
    self.vx *= 0.85
    self.x += self.vx
    if (self.x < 5 and self.vx < 0) or (self.x > 95 and self.vx > 0):
      self.vx *= -0.5
    reward = 0
    live_count = len(self.balls)
    remaining = []
    for ball in self.balls:
      falling = ball.vy >= 0
      ball.x += ball.vx
      ball.y += ball.vy
      ball.vx *= 0.99
      ball.vy = ball.vy * 0.99 + 0.02
      if (ball.x < 0 and ball.vx < 0) or (ball.x > 100 and ball.vx > 0):
        ball.vx *= -1
      if falling and self.pressed and 82 < ball.y < 90 and abs(ball.x - self.x) < 10:
        ball.vy = -2
        ball.vx = (ball.x - self.x) / 10 + self.random.uniform(-0.1, 0.1)
        reward += live_count
      if ball.y > 100:
        live_count -= 1
      else:
        remaining.append(ball)
    self.balls = remaining
    self.spawn_timer -= 1
    if self.spawn_timer <= 0:
      self.balls.append(Ball(self.random.uniform(10, 90)))
      self.spawn_timer = 200
    self.terminal = not self.balls
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    col = min(9, max(0, int(self.x // 10)))
    state[9, col, 0] = True
    for ball in self.balls:
      if 0 <= ball.x < 100 and 0 <= ball.y < 100:
        state[int(ball.y // 10), int(ball.x // 10), 1 if ball.vy >= 0 else 2] = True
    nearest = self._nearest()
    if nearest is not None and 0 <= nearest.x < 100 and 0 <= nearest.y < 100:
      state[int(nearest.y // 10), int(nearest.x // 10), 3] = True
    state[8, max(0, col - 1):col + 2, 4] = True
    values = (self.steps, self.spawn_timer, round((self.vx + 3) * 80), round(self.x * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
