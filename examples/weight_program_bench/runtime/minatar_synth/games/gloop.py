import math

import numpy as np


class GloopEnvironment:
  score_cap = 25

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 50.0
    self.vy = 0.0
    self.gravity = -1
    self.steps = self.action = 0
    self.terminal = False
    self.orbs = [self._point() for _ in range(7)]
    self.mines: list[tuple[float, float]] = []

  def _point(self) -> tuple[float, float]:
    x, y = self.random.uniform(10, 90, 2)
    return x, y

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    dx = {1: -2.5, 3: 2.5}.get(self.action, 0)
    self.x = min(92, max(8, self.x + dx))
    self.vy = (self.vy + 0.02 * self.gravity) * 0.99
    self.y = (self.y + self.vy) % 99
    collision_y = self.y
    while len(self.orbs) < 7:
      self.orbs.append(self._point())
    if self.random.random() < 0.02:
      x, y = self._point()
      if abs(x - self.x) + abs((y - self.y + 50) % 100 - 50) > 25:
        self.mines.append((x, y))
    surviving_orbs = []
    removed = 0
    for x, y in self.orbs:
      if abs(x - self.x) >= 5 or abs(y - collision_y) >= 4:
        surviving_orbs.append((x, y))
        continue
      if abs(self.vy) > 1:
        remaining = [mine for mine in self.mines if math.dist(mine, (x, y)) >= 20]
        removed += len(self.mines) - len(remaining)
        self.mines = remaining
      else:
        surviving_orbs.append((x, y))
      self.vy *= -0.3
      self.gravity *= -1
      self.y = y + self.gravity * 5
    self.orbs = surviving_orbs
    self.terminal = any(abs(x - self.x) < 4 and abs(y - collision_y) < 3 for x, y in self.mines)
    return removed * (removed + 1) // 2

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = int(self.y) // 10, int(self.x) // 10
    state[row, col, 0] = True
    for channel, points in ((1, self.orbs), (2, self.mines)):
      for x, y in points:
        state[int(y) // 10, int(x) // 10, channel] = True
    if abs(self.vy) > 1:
      for x, y in self.orbs:
        state[int(y) // 10, int(x) // 10, 3] = True
    state[int((self.y + 10 * self.vy) % 99) // 10, col, 4] = True
    for row, value in enumerate((self.steps, round((self.vy + 2) * 100), self.action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.gravity > 0
    return state
