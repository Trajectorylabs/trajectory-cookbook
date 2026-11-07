from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


HORIZON = 400
TIME_LIMIT = 1000
FRAMES_PER_STEP = 4


@dataclass
class Barge:
  x: float
  width: float
  velocity: float
  phase: float
  frequency: float
  amplitude: float
  top: float = 65


class BargeHopEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = self.level = 0
    self.pressed = self.terminal = False
    self.remaining = float(TIME_LIMIT)
    self._setup_round()

  def _setup_round(self) -> None:
    speed = 0.175 * (1 + math.sqrt(self.level * 0.3))
    self.barges = [
      Barge(
        center + self.random.uniform(-4, 4),
        float(self.random.integers(20, 29)),
        float(self.random.choice((-1, 1))) * speed * self.random.uniform(0.9, 1.5),
        self.random.uniform(0, math.tau),
        self.random.uniform(0.02, 0.05),
        self.random.uniform(1, 2),
      )
      for center in (32, 75, 118)
    ]
    self._return_to_dock()

  def _return_to_dock(self) -> None:
    self.x, self.feet = 8.0, 63.0
    self.vx = self.vy = 0.0
    self.grounded = True
    self.support: Barge | None = None

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    jump = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(jump and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _move_barges(self) -> None:
    for barge in self.barges:
      barge.x += barge.velocity
      barge.top = 65 - barge.amplitude * (
        1 - math.sin(self.frame * barge.frequency + barge.phase)
      )
      low, high = 15 + barge.width / 2, 135 - barge.width / 2
      if barge.x <= low:
        barge.x, barge.velocity = low, abs(barge.velocity)
      elif barge.x >= high:
        barge.x, barge.velocity = high, -abs(barge.velocity)
    for first, second in combinations(self.barges, 2):
      gap = (first.width + second.width) / 2 - (second.x - first.x)
      if gap <= 0:
        continue
      first.x -= gap / 2
      second.x += gap / 2
      if first.velocity > 0:
        first.velocity *= -1
      if second.velocity < 0:
        second.velocity *= -1

  def _advance_frame(self, jump: bool) -> int:
    difficulty = 1 + self.level / 5
    self.frame += 1
    self.remaining -= math.sqrt(difficulty)
    if self.remaining <= 0:
      self.terminal = True
      return 0
    old_support_x = self.support.x if self.support is not None else 0
    self._move_barges()
    if self.grounded:
      if self.support is not None:
        self.x += self.support.x - old_support_x
        self.feet = self.support.top
      if not jump:
        return 0
      self.grounded = False
      self.support = None
      self.vx = 0.7 * math.sqrt(difficulty)
      self.vy = -2.1 * math.sqrt(difficulty)
    previous_feet = self.feet
    self.vy += 0.15 * (0.5 if self.pressed else 1.5) * difficulty
    self.x += self.vx
    self.feet += self.vy
    if self.vy < 0:
      return 0
    if self.x >= 133 and previous_feet <= 63 <= self.feet:
      reward = int(120 * self.remaining / TIME_LIMIT)
      self.level += 1
      self.remaining = min(TIME_LIMIT, self.remaining + TIME_LIMIT / 2)
      self._setup_round()
      return reward
    for barge in self.barges:
      if abs(self.x - barge.x) <= barge.width / 2 + 3 and previous_feet <= barge.top <= self.feet:
        self.support = barge
        self.grounded = True
        self.feet = barge.top
        self.vx = self.vy = 0
        return 1
    if self.feet >= 71:
      self._return_to_dock()
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    column = min(9, int(self.x / 15))
    row = int((self.feet - 4) / 10)
    state[row, column, 0] = True
    state[row, column, 4] = self.grounded
    state[6, (0, 9), 1] = True
    state[7:9, :, 2] = True
    for barge in self.barges:
      left = max(0, int((barge.x - barge.width / 2) / 15))
      right = min(9, int((barge.x + barge.width / 2) / 15))
      state[int(barge.top / 10), left:right + 1, 3] = True
      state[int(barge.top / 10), left:right + 1, 5] |= barge.velocity < 0
    state[9, :math.ceil(max(0, self.remaining) / 100), 5] = True
    state[8, 0, 5] = self.pressed
    return state
