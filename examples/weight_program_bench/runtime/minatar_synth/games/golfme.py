import math

import numpy as np


class GolfmeEnvironment:
  score_cap = 2400

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 85.0
    self.vx = self.vy = self.angle = 0.0
    self.platform_end = self.gap = 0.0
    self.steps = 0
    self.pressed = self.airborne = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, released: bool) -> float:
    next_start = self.platform_end + self.gap
    if next_start < 0:
      self.platform_end = 200
      self.gap = self.random.uniform(50, 150)
      next_start = self.platform_end + self.gap
    if self.x < 0 or self.y > 99:
      self.terminal = True
      return 0.0
    if self.airborne:
      self.x += self.vx
      self.y += self.vy
      self.vy += 0.1
      on_platform = self.x - 4.5 < self.platform_end or self.x + 4.5 > next_start
      if self.vy > 0 and 85.5 < self.y < 103.5 and on_platform:
        self.airborne = False
        self.angle = 0
        self.y = 85
    else:
      if self.pressed:
        self.angle -= 0.05
      if released:
        self.airborne = True
        self.vx = 4 * math.cos(self.angle)
        self.vy = 4 * math.sin(self.angle)
    scroll = 1 + min(99, max(0, self.x - 50)) * 0.1
    self.x -= scroll
    self.platform_end -= scroll
    return scroll

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y) // 10))
    col = min(9, max(0, int(self.x) // 20))
    state[row, col, 0] = True
    next_start = self.platform_end + self.gap
    for column in range(10):
      left, right = 20 * column, 20 * (column + 1)
      state[9, column, 1] = left < self.platform_end or right > next_start
      state[9, column, 2] = left < next_start and right > self.platform_end
    if not self.airborne:
      for distance in range(5, 25, 5):
        x = self.x + distance * math.cos(self.angle)
        y = self.y + distance * math.sin(self.angle)
        if 0 <= x < 200 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 20, 3] = True
    else:
      x = self.x + 4 * self.vx
      y = self.y + 4 * self.vy
      if 0 <= x < 200 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 20, 4] = True
    for row, value in enumerate((self.steps, round((-self.angle % math.tau) * 70))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.airborne
    return state
