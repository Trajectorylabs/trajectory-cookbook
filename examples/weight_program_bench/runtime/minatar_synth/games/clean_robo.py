from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Wall:
  side: int
  distance: float = 50
  angle: float = 0

  def vertices(self) -> list[tuple[float, float]]:
    angles = [self.angle + (math.pi if self.side < 0 else 0)
              + self.side * offset * math.tau / 3 for offset in (-1, 0, 1)]
    return sorted(
      [(100 + self.side * self.distance + 17 * math.cos(angle),
        50 + 17 * math.sin(angle)) for angle in angles],
      key=lambda point: point[1],
    )

  def edge_x(self, y: float) -> float:
    first, middle, last = self.vertices()
    if y <= first[1]:
      return first[0]
    if y >= last[1]:
      return last[0]
    a, b = (first, middle) if y < middle[1] else (middle, last)
    return a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])

  def target(self) -> tuple[float, float]:
    angle = self.angle + (math.pi if self.side < 0 else 0)
    return (100 + self.side * (self.distance - 7) + 17 * math.cos(angle),
            50 + 8.5 * math.sin(angle))


class CleanRoboEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.x, self.angle, self.speed, self.multiplier = 90.0, 0.0, 1.0, 1.0
    self.direction = 1
    self.walls = {side: Wall(side, angle=self.random.uniform(-math.pi / 7, math.pi / 7))
                  for side in (-1, 1)}
    self.garbage: tuple[float, float] | None = self.walls[1].target()
    self.missed: tuple[float, float] | None = None

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

  def _triangle(self) -> list[tuple[float, float]]:
    return [(self.x + 10 * math.cos(self.angle + index * math.tau / 3),
             50 + 10 * math.sin(self.angle + index * math.tau / 3)) for index in range(3)]

  def _advance_frame(self) -> int:
    if self.pressed:
      self.speed += 0.3
    else:
      self.angle = (self.angle + 0.05) % math.tau
      self.speed += (1 - self.speed) * 0.2
      self.multiplier = max(1, self.multiplier - 0.1)
    old_x = self.x
    self.x += self.direction * self.speed * 0.2
    triangle = self._triangle()
    wall = self.walls[self.direction]
    reward = 0
    if self.garbage is not None:
      gx, gy = self.garbage
      for a, b in zip(triangle, triangle[1:] + triangle[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = min(1, max(0, ((gx - a[0]) * dx + (gy - a[1]) * dy) / 300))
        edge_x = a[0] + dx * fraction
        lo, hi = sorted((edge_x, edge_x + old_x - self.x))
        distance_x = max(lo - gx, gx - hi, 0)
        if math.hypot(distance_x, gy - a[1] - dy * fraction) <= 3:
          reward = math.ceil(self.multiplier)
          self.multiplier += 10
          self.garbage = None
          break

    if (self.x - 100) * self.direction > 0 and any(
      (x - wall.edge_x(y)) * self.direction >= 0 for x, y in triangle
    ):
      self.direction *= -1
      self.missed = self.garbage
      if self.missed is not None:
        self.speed = 1
      wall.distance = min(100, max(30, wall.distance + (20 if self.missed is not None else -10)))
      wall.angle = self.random.uniform(-math.pi / 7, math.pi / 7)
      self.garbage = self.walls[self.direction].target()
      self.terminal = not 0 <= wall.target()[0] + wall.side * 7 <= 199
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, min(9, max(0, int(self.x) // 20)), 0] = True
    for wall in self.walls.values():
      for row in range(10):
        col = min(9, max(0, int(wall.edge_x(row * 10 + 5)) // 20))
        state[row, col, 1] = True
    for channel, point in ((2, self.garbage), (3, self.missed)):
      if point is not None:
        x, y = point
        state[int(y) // 10, min(9, max(0, int(x) // 20)), channel] = True
    for x, y in self._triangle():
      state[int(y) // 10, min(9, max(0, int(x) // 20)), 4] = True
    for row, value in enumerate((self.steps, math.ceil(self.multiplier), round(self.speed * 4), round(self.angle * 40))):
      for bit in range(11):
        if bit < 10:
          state[row, bit, 5] = bool(value & (1 << bit))
        else:
          state[row + 4, 0, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
