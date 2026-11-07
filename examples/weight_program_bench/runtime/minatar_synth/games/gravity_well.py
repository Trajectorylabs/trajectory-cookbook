from dataclasses import dataclass
import math

import numpy as np


@dataclass
class BlackHole:
  x: float
  y: float
  radius: float


class GravityWellEnvironment:
  score_cap = 243

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.y = 50.0
    self.vx = self.vy = 0.0
    self.radius = 9.0
    self.spawn_distance = 0.0
    self.steps = self.frames = 0
    self.pressed = self.terminal = False
    self.holes: list[BlackHole] = []
    self.pulses: list[int] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tapped = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tapped and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool) -> int:
    reward = int(self.radius) if self.frames % 60 == 0 else 0
    self.frames += 1
    for hole in self.holes:
      hole.x -= 0.5
    self.holes = [hole for hole in self.holes if hole.x >= -10]
    self.spawn_distance -= 0.5
    if self.spawn_distance < 0:
      radius = self.random.uniform(5, 9)
      self.holes.append(BlackHole(100 + radius, self.random.uniform(10, 90), radius))
      self.spawn_distance += self.random.uniform(30, 40)
    self.x += self.vx
    self.y += self.vy
    self.vx = (self.vx + self._edge_force(self.x, 110)) * 0.99
    self.vy = (self.vy + self._edge_force(self.y, 105)) * 0.99
    for hole in self.holes:
      dx, dy = hole.x - self.x, hole.y - self.y
      distance_squared = dx * dx + dy * dy
      if distance_squared > 0:
        self.vx += 0.1 * dx / distance_squared
        self.vy += 0.1 * dy / distance_squared
    if tap and self.radius > 2:
      self.radius -= 1
      self.pulses.append(0)
    for radius in self.pulses:
      for hole in self.holes:
        dx, dy = hole.x - self.x, hole.y - self.y
        distance = math.hypot(dx, dy)
        if 0 < distance < radius + 1 + hole.radius:
          force = 0.5 / distance ** 1.5
          self.vx -= dx * force
          self.vy -= dy * force
          hole.x += dx * force
          hole.y += dy * force
    self.pulses = [radius + 1 for radius in self.pulses if radius < 20]
    if any(math.hypot(h.x - self.x, h.y - self.y) < self.radius + h.radius for h in self.holes):
      self.radius -= 0.2
    else:
      self.radius = min(9, self.radius + 0.05)
    self.terminal = self.radius < 1
    if (self.x < 0 and self.vx < 0) or (self.x > 100 and self.vx > 0):
      self.vx *= -1
    if (self.y < 0 and self.vy < 0) or (self.y > 100 and self.vy > 0):
      self.vy *= -1
    return reward

  @staticmethod
  def _edge_force(position: float, far_edge: int) -> float:
    return 1 / max(1, position + 10) ** 2 - 1 / max(1, far_edge - position) ** 2

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y) // 10))
    col = min(9, max(0, int(self.x) // 10))
    state[row, col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for hole in self.holes:
      state[:, :, 1] |= (cols - hole.x) ** 2 + (rows - hole.y) ** 2 < hole.radius ** 2
      if 0 <= hole.x < 100 and 0 <= hole.y < 100:
        state[int(hole.y) // 10, int(hole.x) // 10, 2] = True
    distance = np.hypot(cols - self.x, rows - self.y)
    for radius in self.pulses:
      state[:, :, 3] |= abs(distance - radius) < 5
    future_row = min(9, max(0, int(self.y + 4 * self.vy) // 10))
    future_col = min(9, max(0, int(self.x + 4 * self.vx) // 10))
    state[future_row, future_col, 4] = True
    for gauge_row, value in enumerate((self.steps, round(self.radius * 50))):
      for bit in range(9):
        state[gauge_row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
