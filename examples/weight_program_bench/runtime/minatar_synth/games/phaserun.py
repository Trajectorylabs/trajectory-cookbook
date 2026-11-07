from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Coin:
  x: float
  phase: float = 0

  @property
  def y(self):
    return 50 + 3 * math.sin(self.phase)


class PhaserunEnvironment:
  score_cap = 5000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.size = self.target_size = 8.0
    self.scale_x = 1.0
    self.animation = self.steps = self.frames = 0
    self.obstacle_timer, self.coin_timer = 60, 0
    self.solid = True
    self.pressed = self.terminal = False
    self.obstacles: list[float] = []
    self.coins: list[Coin] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.solid = not self.solid
      self.scale_x = 1.3
      self.animation = 10
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.frames += 1
    if self.animation:
      self.animation -= 1
      self.scale_x += (1 - self.scale_x) * 0.3
    else:
      self.scale_x = 1 + math.sin(self.frames * 0.08) * 0.03
    self.obstacle_timer -= 1
    if self.obstacle_timer <= 0:
      self.obstacles.append(105)
      self.obstacle_timer = int(self.random.integers(88, 111))
    self.coin_timer -= 1
    if self.coin_timer <= 0:
      self.coins.append(Coin(105, self.random.uniform(0, math.tau)))
      self.coin_timer = int(self.random.integers(11, 77))
    self.target_size += 3 if self.pressed else (8 - self.target_size) * 0.5
    self.size += (self.target_size - self.size) * 0.3
    half_width = self.size * self.scale_x / 2
    half_height = self.size * (2 - self.scale_x) / 2
    reward = 0
    remaining = []
    for coin in self.coins:
      coin.x -= 1
      coin.phase += 0.15
      if self.solid and abs(coin.x - 20) < half_width + 3 and abs(coin.y - 50) < half_height + 3:
        reward += math.ceil(self.size)
      elif coin.x >= -10:
        remaining.append(coin)
    self.coins = remaining
    self.obstacles = [x - 1 for x in self.obstacles if x - 1 >= -10]
    self.terminal = self.solid and any(abs(x - 20) < half_width + 5 for x in self.obstacles)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 0 if self.solid else 1] = (
      (abs(cols - 20) <= max(5, self.size * self.scale_x / 2))
      & (abs(rows - 50) <= max(5, self.size * (2 - self.scale_x) / 2))
    )
    for x in self.obstacles:
      if 0 <= x < 100:
        state[5, int(x // 10), 2] = True
    for coin in self.coins:
      if 0 <= coin.x < 100:
        state[int(coin.y // 10), int(coin.x // 10), 3] = True
    state[5, :, 4] = True
    for row, value in enumerate((self.steps, round(self.size), round(self.target_size), self.animation)):
      for bit in range(13):
        gauge_row, col = 2 * row + bit // 10, bit % 10
        state[gauge_row, col, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
