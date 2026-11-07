from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
DIAGONAL = math.sqrt(0.5)


@dataclass
class Bar:
  x: float
  y: float
  width: float
  passed: bool = False


class CatapultEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.passed_count = 0
    self.pressed = self.terminal = False
    self.x, self.y, self.vx, self.vy = 200.0, 60.0, 0.0, 0.0
    self.bar_position, self.base_score, self.next_bar = -10.0, 1.0, 50.0
    self.support: Bar | None = Bar(200, 60, 20, passed=True)
    self.bars = [self.support]

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
    reward = 0
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, tap: bool) -> int:
    scroll = 0.05 + max(0, self.x - 30) * 0.1
    slope = 1 if self.pressed else -1
    self.x -= scroll
    if self.support is not None:
      self.bar_position -= slope * 0.2
      self.base_score = max(1, self.base_score - 0.5)
      self.x = self.support.x + slope * (4 - self.bar_position) * DIAGONAL
      self.y = self.support.y - (4 + self.bar_position) * DIAGONAL
      if abs(self.bar_position) > self.support.width * 0.65:
        self.support = None
        self.vx = self.vy = 0.0
      elif tap and self.bar_position > 0:
        self.vx = math.sqrt(self.bar_position) * DIAGONAL
        self.vy = -self.vx
        self.x += self.vx
        self.y += self.vy
        self.support = None
        self.passed_count = 0
    else:
      self.x += self.vx
      self.y += self.vy
      self.vy += 0.1
      self.base_score += self.vx * 0.1

    if self.y > 99 or self.x < 0:
      self.terminal = True
      return 0

    reward = 0
    for bar in self.bars:
      bar.x -= scroll
      dx, dy = self.x - bar.x, self.y - bar.y
      if self.support is None and abs(dy - slope * dx) * DIAGONAL < 4 and abs(dx + slope * dy) * DIAGONAL < bar.width / 2 + 2:
        self.support = bar
        offset = math.hypot(dx - slope * 4 * DIAGONAL, dy + 4 * DIAGONAL)
        self.bar_position = max(-bar.width * 0.4, min(bar.width * 0.4, offset if dx <= slope * 4 * DIAGONAL else -offset))
        reward += math.ceil(self.base_score)
        bar.passed = True
      if not bar.passed and bar.x + bar.width / 2 < self.x:
        bar.passed = True
        self.passed_count += 1
        bonus = 10 * self.passed_count
        reward += bonus
        self.base_score += bonus
    self.bars = [bar for bar in self.bars if bar.x >= -bar.width / 2]

    self.next_bar -= scroll
    if self.next_bar < 0:
      width = self.random.uniform(20, 30)
      self.bars.append(Bar(200 + width / 2, self.random.uniform(50, 90), width))
      self.next_bar += width / 2 + self.random.uniform(30, 50)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y) // 10))
    column = min(9, max(0, int(self.x) // 20))
    state[row, column, 0] = True
    state[row, column, 3] = self.support is not None
    slope = 1 if self.pressed else -1
    for bar in self.bars:
      for distance in np.linspace(-bar.width / 2, bar.width / 2, 9):
        x = bar.x + distance * DIAGONAL
        y = bar.y + slope * distance * DIAGONAL
        if 0 <= x < 200 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 20, 1 if bar.passed else 2] = True
    for row, value in enumerate((self.steps, round((self.bar_position + 32) * 4), math.ceil(self.base_score), self.passed_count)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 4] = self.pressed, self.vy < 0, self.bar_position > 0
    return state
