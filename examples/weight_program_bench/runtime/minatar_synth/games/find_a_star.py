from dataclasses import dataclass

import numpy as np


@dataclass
class BoxLine:
  y: float
  secret: int
  low: int = 0
  high: int = 15


@dataclass
class Star:
  y: float
  column: int
  score: int
  timer: int = 32
  velocity: float = 1


class FindAStarEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.cursor = 7
    self.scroll_boost = self.spawn_distance = 0.0
    self.terminal = False
    self.lines: list[BoxLine] = []
    self.stars: list[Star] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    tap = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    if action in (1, 4):
      self.cursor = min(15, max(0, self.cursor + (-1 if action == 1 else 1)))
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool) -> int:
    if tap:
      self.scroll_boost += 2
    scroll = 0.06 + self.scroll_boost
    self.scroll_boost *= 0.07
    self.spawn_distance -= scroll
    if self.spawn_distance < 0:
      self.lines.append(BoxLine(-3, int(self.random.integers(16))))
      self.spawn_distance += 10

    remaining = []
    combo = 1
    for index, line in enumerate(self.lines, 1):
      if line.y < 9:
        line.y += (9 - line.y) * 0.2
      elif tap:
        if self.cursor == line.secret:
          self.stars.append(Star(line.y, self.cursor, index**2 * combo))
          combo += 1
          continue
        if self.cursor < line.secret:
          line.low = max(line.low, self.cursor + 1)
        else:
          line.high = min(line.high, self.cursor - 1)
      line.y += scroll
      self.terminal |= line.y > 97
      remaining.append(line)
    self.lines = remaining
    if not self.lines:
      self.spawn_distance = 0

    reward = 0
    active = []
    for star in self.stars:
      star.y += star.velocity
      star.velocity *= 0.9
      star.timer -= 1
      if star.timer:
        active.append(star)
      else:
        reward += star.score
    self.stars = active
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for line in self.lines:
      if 0 <= line.y < 100:
        row = int(line.y) // 10
        for column in range(16):
          half, slot = divmod(column, 8)
          state[row, slot, 2 * half + int(not line.low <= column <= line.high)] = True
    for star in self.stars:
      if 0 <= star.y < 100:
        state[int(star.y) // 10, star.column // 2, 4] = True
    half, slot = divmod(self.cursor, 8)
    state[half, slot, 5] = True
    for bit in range(9):
      state[9, bit, 5] = bool(self.steps & (1 << bit))
    state[8, 9, 5] = self.last_action == 5
    return state
