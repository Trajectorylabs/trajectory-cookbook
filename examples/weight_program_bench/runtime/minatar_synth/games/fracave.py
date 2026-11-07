from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Wall:
  x: float
  y: float
  dx: float
  dy: float

  def near(self, x: float, y: float) -> bool:
    dx, dy = x - self.x, y - self.y
    length = self.dx**2 + self.dy**2
    fraction = min(1, max(0, (dx * self.dx + dy * self.dy) / length)) if length else 0
    return math.hypot(dx - fraction * self.dx, dy - fraction * self.dy) < 4


class FracaveEnvironment:
  score_cap = 50

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = 0
    self.x, self.y, self.angle, self.speed = 40.0, 75.0, 0.0, 1.0
    self.cooldown = 0.0
    self.pressed = self.terminal = False
    self.walls: list[Wall] = []
    self._generate(0)

  def _generate(self, y: float) -> None:
    for x in (20, 80):
      self._subdivide(x, y, x, y + 150, 3)

  def _subdivide(self, x1: float, y1: float, x2: float, y2: float, depth: int) -> None:
    if not depth:
      self.walls.append(Wall(x1, y1, x2 - x1, y2 - y1))
      return
    x = (x1 + x2) / 2 + self.random.uniform(-10, 10)
    y = (y1 + y2) / 2 + self.random.uniform(-5, 5)
    self._subdivide(x1, y1, x, y, depth - 1)
    self._subdivide(x, y, x2, y2, depth - 1)

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
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    if self.frame % 150 == 0:
      self._generate(-150)
    self.frame += 1
    self.speed += ((2 if self.pressed else 0.1) - self.speed) * 0.3
    self.x += math.cos(self.angle) * self.speed
    self.y += math.sin(self.angle) * self.speed
    self.cooldown -= self.speed
    for wall in self.walls:
      wall.y += 1
    self.walls = [wall for wall in self.walls if wall.y < 170]
    reward = 0
    if self.cooldown < 0:
      for wall in self.walls:
        if wall.near(self.x, self.y):
          tangent = math.atan2(wall.dy, wall.dx)
          self.angle = (2 * tangent - self.angle) % math.tau
          self.x += math.cos(self.angle) * 7
          self.y += math.sin(self.angle) * 7
          self.speed, self.cooldown = 0.2, 9
          reward = 1
          break
    self.terminal = not (0 <= self.x <= 100 and 0 <= self.y <= 150)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)

    def mark(x: float, y: float, channel: int) -> None:
      if 0 <= x <= 100 and 0 <= y <= 150:
        state[min(9, int(y) // 15), min(9, int(x) // 10), channel] = True

    mark(self.x, self.y, 0)
    mark(self.x + 9 * math.cos(self.angle), self.y + 9 * math.sin(self.angle), 2)
    if self.cooldown > 0:
      state[:, :, 3] = state[:, :, 0]
    for wall in self.walls:
      for fraction in np.linspace(0, 1, 12):
        mark(wall.x + wall.dx * fraction, wall.y + wall.dy * fraction, 1)
    state[(0, 9), :, 4] = True
    state[:, (0, 9), 4] = True
    for row, value in enumerate((self.steps, round(self.angle * 80), round(self.speed * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
