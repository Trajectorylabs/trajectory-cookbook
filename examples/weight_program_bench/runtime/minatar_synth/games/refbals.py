from dataclasses import dataclass

import numpy as np


@dataclass
class Ball:
  x: float
  y: float = 0
  velocity: float = 0


class RefbalsEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.steps = self.frame = 0
    self.held = self.terminal = False
    self.balls = []
    self.walls = [(-29 * i, -9) for i in range(5)]

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.frame % 99 == 0:
      self.balls.append(Ball(self.random.uniform(0, 50)))
    moving_walls = [(x - (2 if self.held else 1), y) for x, y in self.walls]
    self.walls = [
      (x + self.random.uniform(130, 150), self.random.uniform(50, 90))
      if x < -19 else (x, y)
      for x, y in moving_walls
    ]
    reward = 0
    for ball in self.balls:
      ball.velocity += 0.03
      ball.y += ball.velocity
      self.terminal |= ball.y > 99
      if any(abs(ball.x - x) < 20.5 and abs(ball.y - y) < 4 for x, y in moving_walls):
        reward += 1
        ball.velocity *= -1
        ball.y += ball.velocity * 2
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[:100:10, :100:10]
    for x, y in self.walls:
      state[:, :, 2] |= (cols < x + 18) & (cols + 10 > x - 18) & (rows < y + 1.5) & (rows + 10 > y - 1.5)
    for ball in self.balls:
      row, col = int(np.clip(ball.y // 10, 0, 9)), int(ball.x // 10)
      state[row, col, int(ball.velocity < 0)] = True
      if ball.y < 0:
        state[0, col, 3] = True
      if ball.y >= 80 and ball.velocity > 0:
        state[row, col, 4] = True
    for row, value in enumerate((self.steps, self.frame % 99, self.held)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
