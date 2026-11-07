from dataclasses import dataclass

import numpy as np


@dataclass
class Hazard:
  x: float
  y: float
  direction: int
  snake: bool


class HoppinHazardsEnvironment:
  score_cap = 318.1

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.y, self.size = 90, 10
    self.spawn_timer = 0.0
    self.steps = 0
    self.pressed = self.terminal = False
    self.hazards: list[Hazard] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    cents = 0
    for _ in range(4):
      cents += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return cents / 100, self.terminal

  def _advance(self) -> int:
    rise = 1 if self.pressed else -2
    self.y = min(90, max(10, self.y - rise))
    self.size = min(30, max(10, self.size + rise))
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      direction = 1 if self.random.integers(2) == 0 else -1
      snake = bool(self.random.integers(2))
      self.hazards.append(Hazard(-10 if direction == 1 else 110,
                                 self.random.uniform(10, 90), direction, snake))
      self.spawn_timer = self.random.uniform(45, 120)
    half_size = self.size / 2
    for hazard in self.hazards:
      hazard.x += hazard.direction
      if hazard.snake:
        # Use the bounding box of the leading semicircle.
        center = hazard.x + 2.5 * hazard.direction
        hit = abs(center - 50) < half_size + 4 and abs(hazard.y - self.y) < half_size + 6.5
      else:
        hit = abs(hazard.x - 50) < half_size + 5 and abs(hazard.y - self.y) < half_size + 1.5
      self.terminal |= hit
    self.hazards = [hazard for hazard in self.hazards if -10 <= hazard.x <= 110]
    return self.size - 10

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    top = max(0, int((self.y - self.size / 2) // 10))
    bottom = min(9, int((self.y + self.size / 2) // 10))
    left = int((50 - self.size / 2) // 10)
    right = int((50 + self.size / 2) // 10)
    state[top:bottom + 1, left:right + 1, 0] = True
    for hazard in self.hazards:
      if 0 <= hazard.x < 100:
        row, col = int(hazard.y // 10), int(hazard.x // 10)
        state[row, col, 1 + 2 * hazard.snake + (hazard.direction < 0)] = True
    for row, value in enumerate((self.steps, self.y, self.size)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
