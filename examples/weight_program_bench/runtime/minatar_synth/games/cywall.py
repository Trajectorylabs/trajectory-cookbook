from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Ring:
  x: float
  y: float
  radius: float
  count: int
  angle: float
  spin: float
  length: float
  following: "Ring | None" = None

  def walls(self):
    for index in range(self.count):
      angle = self.angle + index * math.tau / self.count
      cos, sin = math.cos(angle), math.sin(angle)
      yield self.x + cos * self.radius, self.y + sin * self.radius, -sin, cos


class CywallEnvironment:
  score_cap = 1000

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
    self.rings: list[Ring] = []
    self.player = self._spawn()
    self.next_ring = self.random.uniform(20, 40)

  def _spawn(self) -> Ring:
    radius = self.random.uniform(20, 30)
    ring = Ring(self.random.uniform(20, 80), -radius, radius,
                int(self.random.integers(1, 4)), self.random.uniform(0, math.tau),
                self.random.uniform(0.02, 0.05) * self.random.choice((-1, 1)),
                self.random.uniform(10, 20))
    if self.rings:
      self.rings[-1].following = ring
    self.rings.append(ring)
    return ring

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for frame in range(4):
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance_frame(self, tap: bool) -> float:
    if self.next_ring <= 0:
      self._spawn()
      self.next_ring += self.random.uniform(20, 40)
    scroll = 0.1 + max(0, 50 - self.player.y) * 0.05
    self.next_ring -= scroll
    for ring in self.rings:
      ring.y += scroll
      ring.angle = (ring.angle + ring.spin) % math.tau
    self.rings = [ring for ring in self.rings if ring.y <= 99 + ring.radius]
    if self.player.y > 99:
      self.terminal = True
    elif tap and self.player.following is not None:
      target = self.player.following
      if self._blocked(target):
        self.terminal = True
      else:
        self.player = target
    return scroll

  def _blocked(self, target: Ring) -> bool:
    start = self.player
    dx, dy = target.x - start.x, target.y - start.y
    samples = max(1, math.ceil(math.hypot(dx, dy)))
    for ring in self.rings:
      for x, y, wx, wy in ring.walls():
        for index in range(samples + 1):
          fraction = index / samples
          px, py = start.x + dx * fraction - x, start.y + dy * fraction - y
          if abs(px * wx + py * wy) <= ring.length / 2 + 1.5 and abs(px * wy - py * wx) <= 3:
            return True
      if ring is not start and ring is not target:
        distance_sq = dx * dx + dy * dy
        fraction = min(1, max(0, ((ring.x - start.x) * dx + (ring.y - start.y) * dy) / distance_sq)) if distance_sq else 0
        if math.hypot(start.x + dx * fraction - ring.x, start.y + dy * fraction - ring.y) < 3:
          return True
    return False

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for ring in self.rings:
      col = int(ring.x) // 10
      if 0 <= ring.y < 100:
        channel = 0 if ring is self.player else 1 if ring is self.player.following else 2
        state[int(ring.y) // 10, col, channel] = True
      for x, y, wx, wy in ring.walls():
        for offset in np.linspace(-ring.length / 2, ring.length / 2, 9):
          px, py = x + offset * wx, y + offset * wy
          if 0 <= px < 100 and 0 <= py < 100:
            state[int(py) // 10, int(px) // 10, 3] = True
    if self.player.following is not None:
      target = self.player.following
      for fraction in np.linspace(0, 1, 16):
        x = self.player.x + fraction * (target.x - self.player.x)
        y = self.player.y + fraction * (target.y - self.player.y)
        if 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, 4] = True
    for row, value in enumerate((self.steps, round(self.player.y + 32))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
