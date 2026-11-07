from dataclasses import dataclass

import numpy as np


DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))
HORIZON = 400


@dataclass(eq=False)
class Point:
  x: float
  y: float
  direction: int

  def cell(self) -> tuple[int, int]:
    return int(self.y // 10), int(self.x // 10)


@dataclass
class Line:
  start: Point
  end: Point
  age: int = 0

  def crosses(self, other: "Line") -> bool:
    def side(a: Point, b: Point, p: Point) -> float:
      return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)

    return (
      side(self.start, self.end, other.start) * side(self.start, self.end, other.end) < 0
      and side(other.start, other.end, self.start) * side(other.start, other.end, self.end) < 0
    )


class CrossLineEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.spawn_count = 0
    self.next_point = 0.0
    self.terminal = False
    self.cursor = (5, 5)
    self.selected = Point(70, 1, 1)
    self.points = [self.selected]
    self.lines = [Line(Point(30, 99, -1), self.selected)]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    if 1 <= action <= 4:
      dy, dx = DIRECTIONS[action - 1]
      row, col = self.cursor
      self.cursor = min(9, max(0, row + dy)), min(9, max(0, col + dx))
    connect = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance_frame(connect and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, connect: bool) -> int:
    depth = max((point.y if point.direction > 0 else 100 - point.y for point in self.points), default=0)
    scroll = 0.03 + max(0, 40 - depth) * 0.1
    self.next_point -= scroll
    while self.next_point < 0:
      direction = -1 if self.spawn_count % 2 == 0 else 1
      pair = self.spawn_count // 2 % 7
      lo, hi = {3: (10, 30), 6: (70, 90)}.get(pair, (10, 90))
      y = -19 - self.next_point if direction > 0 else 119 + self.next_point
      self.points.append(Point(self.random.uniform(lo, hi), y, direction))
      self.next_point += 4
      self.spawn_count += 1

    endpoints = {point for line in self.lines for point in (line.start, line.end)}
    for point in endpoints.union(self.points):
      point.y += scroll * point.direction
    reward = self._connect() if connect else 0
    self.terminal = any(
      not 0 <= point.y <= 99
      for line in self.lines for point in (line.start, line.end)
    )
    return reward

  def _connect(self) -> int:
    candidate = next((point for point in self.points
                      if point is not self.selected and point.cell() == self.cursor), None)
    if candidate is None:
      return 0
    new_line = Line(self.selected, candidate)
    self.points.remove(self.selected)
    self.selected = candidate
    reward, multiplier = 0, 1
    survivors = []
    for line in self.lines:
      line.age += 1
      if line.age >= 2 and line.crosses(new_line):
        reward += multiplier
        multiplier *= 2
      else:
        survivors.append(line)
    self.lines = [*survivors, new_line]
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for point in self.points:
      row, col = point.cell()
      if 0 <= row < 10 and 0 <= col < 10:
        channel = 2 if point is self.selected else int(point.direction < 0)
        state[row, col, channel] = True
    for line in self.lines:
      for fraction in np.linspace(0, 1, 20):
        row = int((line.start.y + (line.end.y - line.start.y) * fraction) // 10)
        col = int((line.start.x + (line.end.x - line.start.x) * fraction) // 10)
        if 0 <= row < 10 and 0 <= col < 10:
          state[row, col, 3] = True
          state[row, col, 4] |= line.age >= 1
    row, col = self.cursor
    state[row, col, 5] = True
    return state
