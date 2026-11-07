from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Ring:
  x: float
  y: float
  radius: float
  danger: bool = False


class CircleWEnvironment:
  score_cap = 25000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.player = Ring(200, 50, 1)
    self.scroll_x = self.scroll_y = self.next_ring = 0.0
    self.danger_count = 30
    self.multiplier = 1
    self.rings: list[Ring] = []

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
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  @staticmethod
  def _touches(first: Ring, second: Ring) -> bool:
    distance = math.hypot(first.x - second.x, first.y - second.y)
    return abs(first.radius - second.radius) - 2 <= distance <= first.radius + second.radius + 2

  def _advance_frame(self) -> int:
    player = self.player
    if player.radius < 9:
      self.multiplier = max(1, self.multiplier - 1)
    player.radius = max(1, player.radius + (0.5 if self.pressed else -0.5))
    for ring in [player, *self.rings]:
      ring.x += self.scroll_x
      ring.y = (ring.y + self.scroll_y + 15) % 130 - 15
    player.x = max(20, player.x)
    self.rings = [ring for ring in self.rings if ring.x >= -ring.radius]
    destination = None
    for ring in self.rings:
      if self._touches(player, ring):
        if ring.danger:
          self.terminal = True
          return 0
        if destination is None:
          destination = ring
    reward = 0
    if destination is not None:
      self.multiplier += math.ceil(player.radius)
      reward = self.multiplier
      self.player = destination
      self.rings.remove(destination)

    self.next_ring += self.scroll_x
    if self.next_ring < 0:
      self._spawn()
    self.scroll_x = -1 - max(0, player.x - 20) * 0.1
    self.scroll_y = (50 - player.y) * 0.1
    return reward

  def _spawn(self) -> None:
    danger = self.danger_count == 0
    for _ in range(9):
      radius = self.random.uniform(8, 15)
      x, y = 200 + radius, int(self.random.integers(99))
      half_width = radius * (4 if danger else 2.5)
      if any(abs(x - ring.x) < half_width + ring.radius and abs(y - ring.y) < radius * 1.25 + ring.radius
             for ring in self.rings):
        continue
      self.rings.append(Ring(x, y, radius, danger))
      self.danger_count = int(self.random.integers(24, 30)) if danger else self.danger_count - 1
      self.next_ring = 5
      break

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for ring, channel in [(self.player, 0), *((ring, 2 if ring.danger else 1) for ring in self.rings)]:
      for row in range(10):
        for col in range(10):
          distance = math.hypot(col * 20 + 10 - ring.x, row * 10 + 5 - ring.y)
          state[row, col, channel] |= abs(distance - ring.radius) <= 9
    row = min(9, max(0, int(self.player.y) // 10))
    col = min(9, int(self.player.x) // 20)
    state[row, col, 3] = True
    for ring in self.rings:
      if 0 <= ring.x < 200 and 0 <= ring.y < 100:
        state[int(ring.y) // 10, int(ring.x) // 20, 4] = True
    for row, value in enumerate((self.steps, round(self.player.radius * 2), self.multiplier, self.danger_count)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    for bit in range(6):
      state[4, bit, 5] = bool(self.multiplier & (1 << (bit + 10)))
    state[9, 0, 5] = self.pressed
    return state
