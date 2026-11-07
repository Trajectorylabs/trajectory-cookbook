from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Platform:
  x: float
  center: float
  angle: float
  radius: float
  speed: float
  used: bool = False

  @property
  def y(self) -> float:
    return self.center + math.sin(self.angle) * self.radius


class FloaterEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.x, self.y, self.vx, self.vy = 105.0, 10.0, 0.0, 0.0
    self.pressed = self.terminal = False
    self.next_platform = 0.0
    self.center = 50.0
    self.support: Platform | None = None
    self.platforms: list[Platform] = []

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
    if self.next_platform <= 0:
      radius = self.random.uniform(10, 20)
      low, high = 30 + radius, 80 - radius
      if self.center < low:
        self.center = 2 * low - self.center
      elif self.center > high:
        self.center = 2 * high - self.center
      self.platforms.append(Platform(105, self.center, self.random.uniform(0, math.tau),
                                     radius, self.random.uniform(0.05, 0.1)))
      self.center += self.random.uniform(-20, 20)
      self.next_platform += self.random.uniform(20, 40)

    scroll = 0.1 + max(0, self.x - 30) * 0.05
    self.next_platform -= scroll
    for platform in self.platforms:
      platform.x -= scroll
      platform.angle = (platform.angle + platform.speed) % math.tau
    self.platforms = [platform for platform in self.platforms if platform.x >= -10]
    self.x -= scroll
    if self.support is not None:
      self.y = self.support.y - 6
      if tap:
        self.vx = 1
        self.vy = math.cos(self.support.angle) * self.support.speed * 20 - 1
        self.support.used = True
        self.support = None
    else:
      self.vy += 0.05 if self.pressed else 0.2
      self.x += self.vx
      self.y += self.vy
      if self.y < 8 and self.vy < 0:
        self.vy *= -0.25
      for platform in self.platforms:
        if not platform.used and abs(platform.x - self.x) < 11 and abs(platform.y - self.y) < 6:
          self.support = platform
          break
    self.terminal = self.y > 99 or self.x < 0
    return scroll

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y) // 10))
    col = min(9, max(0, int(self.x) // 10))
    state[row, col, 0] = True
    state[row, col, 3] = self.support is not None
    state[row, col, 4] = self.vy < 0
    for platform in self.platforms:
      if 0 <= platform.y < 100:
        left = max(0, math.floor((platform.x - 7.5) / 10))
        right = min(10, math.ceil((platform.x + 7.5) / 10))
        state[int(platform.y) // 10, left:right, 2 if platform.used else 1] = True
    for row, value in enumerate((self.steps, round(self.vy * 16) + 128)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
