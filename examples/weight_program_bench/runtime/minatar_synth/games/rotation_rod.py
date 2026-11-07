from dataclasses import dataclass
from math import ceil, cos, pi, sin

import numpy as np


@dataclass
class Mover:
  x: float
  y: float
  vx: float
  vy: float

  def advance(self):
    self.x += self.vx
    self.y += self.vy
    return -5 <= self.x <= 110 and -5 <= self.y <= 110


class RotationRodEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = 0.0
    self.direction = 1
    self.steps = self.obstacle_timer = self.coin_timer = 0
    self.multiplier = 1.0
    self.held = self.terminal = False
    self.obstacles, self.coins = [], []
    self.entry = None

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.direction *= -1
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _rod(self):
    distance = np.arange(-24, 25)
    dx, dy = cos(self.angle), sin(self.angle)
    xs, ys = 50 + distance * dx, 50 + distance * dy
    tail = self.angle + 0.2 * self.direction
    arrow_x = np.linspace(50 + 23.04 * dx, 50 + 20.16 * cos(tail), 8)
    arrow_y = np.linspace(50 + 23.04 * dy, 50 + 20.16 * sin(tail), 8)
    return np.concatenate((xs, arrow_x)), np.concatenate((ys, arrow_y))

  def _advance(self):
    if not self.held:
      self.angle = (self.angle + self.direction * 0.05) % (2 * pi)
    xs, ys = self._rod()
    self.obstacle_timer -= 1
    if self.obstacle_timer < 0:
      offset = self.random.uniform(20, 40) + 40 * int(self.random.integers(2))
      entries = ((offset, -3, 0, 0.5), (103, offset, -0.5, 0),
                 (offset, 103, 0, -0.5), (-3, offset, 0.5, 0))
      self.entry = entries[self.random.integers(4)]
      self.obstacles.append(Mover(*self.entry))
      self.obstacle_timer, self.coin_timer = 99, 9
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.advance()]
    self.terminal = any(self._hit(obstacle, xs, ys, 3.5) for obstacle in self.obstacles)
    self.coin_timer -= 1
    if self.coin_timer < 0:
      self.coins.append(Mover(*self.entry))
      self.coin_timer = 9
    reward = 0
    remaining = []
    for coin in self.coins:
      if coin.advance():
        if self._hit(coin, xs, ys, 2.5):
          reward += ceil(self.multiplier)
          self.multiplier += 10
        else:
          remaining.append(coin)
    self.coins = remaining
    self.multiplier *= 0.99
    return reward

  @staticmethod
  def _hit(mover, xs, ys, reach):
    return bool(np.any((abs(xs - mover.x) < reach) & (abs(ys - mover.y) < reach)))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    xs, ys = self._rod()
    state[(ys // 10).astype(int), (xs // 10).astype(int), 0] = True
    for channel, movers in ((1, self.obstacles), (2, self.coins)):
      for mover in movers:
        if 0 <= mover.x < 100 and 0 <= mover.y < 100:
          state[int(mover.y // 10), int(mover.x // 10), channel] = True
    state[int(ys[48] // 10), int(xs[48] // 10), 3] = True
    if self.entry is not None:
      x, y, _, _ = self.entry
      state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), 4] = True
    values = (self.steps, self.held, self.direction > 0, round(self.angle * 100),
              self.obstacle_timer, self.coin_timer, round(self.multiplier))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
