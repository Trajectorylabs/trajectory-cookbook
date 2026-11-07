from dataclasses import dataclass

import numpy as np


@dataclass
class Platform:
  x: float
  y: float
  width: float


class LevitationEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 40.0
    self.vy = self.spawn_distance = 0.0
    self.platforms = [Platform(0, 60, 99)]
    self.steps = 0
    self.multiplier = 1
    self.rolling = self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.rolling = not self.rolling
      self.multiplier = 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.x += 0.2 if self.rolling else -0.4
    self.vy = (self.vy + 0.2) * 0.99 if self.rolling else 0
    self.y += self.vy
    if self.y < 0 and self.vy < 0:
      self.y = 0
      self.vy *= -0.3
    self.spawn_distance -= 0.5
    if self.spawn_distance < 0:
      width = self.random.uniform(30, 50)
      self.platforms.append(Platform(100, self.random.uniform(30, 90), width))
      self.spawn_distance = width + self.random.uniform(0, 9)
    self.terminal = self.y > 100 or not -3 <= self.x <= 103
    if self.terminal:
      return 0
    reward = 0
    for platform in self.platforms:
      if (self.rolling and self.vy >= 0 and platform.x - 3 < self.x < platform.x + platform.width + 3
          and platform.y - 3 < self.y < platform.y + 8):
        reward += self.multiplier
        self.multiplier += 1
        self.y = platform.y - 3
        self.vy *= -1.5
      platform.x -= 0.5
    self.platforms = [p for p in self.platforms if p.x + p.width >= 0]
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    col = min(9, max(0, int(self.x // 10)))
    row = min(9, max(0, int(self.y // 10)))
    state[row, col, int(self.rolling)] = True
    for platform in self.platforms:
      left = max(0, int(platform.x // 10))
      right = min(9, int((platform.x + platform.width) // 10))
      state[int(platform.y // 10), left:right + 1, 2] = True
    future_row = min(9, max(0, int((self.y + 4 * self.vy) // 10)))
    state[future_row, col, 3] = True
    state[:, (0, 9), 4] = True
    gauges = (self.steps, min(511, self.multiplier), min(511, max(0, round((self.vy + 25) * 10))))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
