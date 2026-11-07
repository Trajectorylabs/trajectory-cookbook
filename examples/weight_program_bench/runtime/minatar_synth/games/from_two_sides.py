from dataclasses import dataclass

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Gap:
  x: float
  velocity: float


@dataclass
class Arrow:
  x: int
  y: float
  direction: int
  velocity: float = 0


class FromTwoSidesEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.x = 50
    self.terminal = False
    self.gaps = [Gap(50, -1), Gap(50, 1)]
    self.timers = [0, 60]
    self.arrows: list[Arrow] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    direction = -1 if action == 1 else 1 if action == 4 else 0
    reward = 0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance(direction)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, direction: int) -> int:
    self.x = min(98, max(1, self.x + 2 * direction))
    for gap in self.gaps:
      gap.x += gap.velocity
      if (gap.x < 9 and gap.velocity < 0) or (gap.x > 90 and gap.velocity > 0):
        gap.velocity *= -1
      gap.velocity = (gap.velocity + self.random.uniform(-0.5, 0.5)) * 0.98

    reward = 0
    for index, heading in enumerate((1, -1)):
      self.timers[index] -= 1
      if self.timers[index] < 0:
        width = self.random.uniform(20, 50)
        for x in range(2, 99, 6):
          if all(abs(x - gap.x) >= width / 2 for gap in self.gaps):
            self.arrows.append(Arrow(x, -3 if heading > 0 else 103, heading))
            reward += 1
        self.timers[index] = int(self.random.integers(60, 91))

    survivors = []
    for arrow in self.arrows:
      arrow.velocity += (arrow.direction * 2 - arrow.velocity) * 0.05
      arrow.y += arrow.velocity
      self.terminal |= abs(arrow.x - self.x) < 4 and abs(arrow.y - 50) < 5
      if -3 <= arrow.y <= 103:
        survivors.append(arrow)
    self.arrows = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, self.x // 10, 0] = True
    for arrow in self.arrows:
      if 0 <= arrow.y < 100:
        state[int(arrow.y) // 10, arrow.x // 10, 1 if arrow.direction > 0 else 2] = True
    for index, gap in enumerate(self.gaps):
      state[0, min(9, max(0, int(gap.x) // 10)), 3 + index] = True
    for row, value in enumerate((self.steps, self.x, *self.timers)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
