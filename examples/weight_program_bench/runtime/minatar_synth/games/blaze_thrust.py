from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Bubble:
  x: float
  y: float
  size: float


class BlazeThrustEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.hold_time = 0
    self.pressed = self.terminal = False
    self.x, self.y = 50.0, 90.0
    self.vx = self.vy = 0.0
    self.size = 9.0
    self.multiplier = 1
    self.next_bubble = self.random.uniform(10, 40)
    self.winds = [(0.0, self.random.uniform(20, 40))]
    self.bubbles: list[Bubble] = []

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
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _wind_left(self, y: float) -> bool:
    return any(top <= y <= bottom for top, bottom in self.winds)

  def _advance_frame(self) -> int:
    self.vy += -0.1 if self.pressed else 0.1
    self.hold_time = self.hold_time + 1 if self.pressed else max(0, self.hold_time - 3)
    scroll = max(0, (60 - self.y) * 0.1)
    self.x += self.vx
    self.y += self.vy + scroll
    self.vx *= 0.95
    self.vy *= 0.95
    if self.x < 5:
      self.x, self.vx = 10, abs(self.vx) * 0.5
    elif self.x > 95:
      self.x, self.vx = 90, -abs(self.vx) * 0.5
    if self.y > 95:
      self.y, self.vy = 90, -abs(self.vy) * 0.3

    if scroll:
      self.winds = [(top + scroll, bottom + scroll) for top, bottom in self.winds if top + scroll < 120]
      top = self.winds[0][0]
      if top >= 0:
        bottom = top - self.random.uniform(20, 40)
        self.winds.insert(0, (bottom - self.random.uniform(20, 40), bottom))
    self.vx += self.size * (-0.02 if self._wind_left(self.y) else 0.02)
    self.size *= 1 - 0.0016 * (1 + self.hold_time * 0.01)
    if self.size <= 2:
      self.terminal = True
      return 0

    reward = self._advance_bubbles(scroll)
    if scroll and len(self.bubbles) < 15:
      self.next_bubble -= scroll
      if self.next_bubble <= 0:
        size = self.random.uniform(3, 9)
        self.bubbles.append(Bubble(self.random.uniform(10, 90), -size, size))
        self.next_bubble = self.random.uniform(5, 55)
    return reward

  def _advance_bubbles(self, scroll: float) -> int:
    reward = 0
    remaining = []
    for bubble in self.bubbles:
      bubble.y += scroll + 0.3 - (bubble.size - 3) * 0.015
      bubble.x += bubble.size * (-0.02 if self._wind_left(bubble.y) else 0.02)
      bubble.x = min(95, max(5, bubble.x))
      if math.hypot(bubble.x - self.x, bubble.y - self.y) <= bubble.size + self.size:
        reward += self.multiplier
        self.multiplier = min(16, self.multiplier + 1)
        self.size = min(25, self.size + bubble.size * 0.2)
      elif bubble.y > 120:
        self.multiplier = 1
      else:
        remaining.append(bubble)
    self.bubbles = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, column = int(self.y) // 10, int(self.x) // 10
    state[row, column, 0] = True
    state[row, column, 4] = self.vy < 0
    for row in range(10):
      state[row, :, 1 if self._wind_left(10 * row + 5) else 2] = True
    for bubble in self.bubbles:
      if 0 <= bubble.y < 100:
        state[int(bubble.y) // 10, int(bubble.x) // 10, 3] = True
    for row, value in enumerate((round(self.size * 10), self.multiplier, self.hold_time, self.steps)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
      state[row + 4, 0, 5] = bool(value & 1024)
    state[9, 0, 5] = self.pressed
    return state
