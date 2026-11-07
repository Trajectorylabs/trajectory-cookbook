from dataclasses import dataclass
import math

import numpy as np


@dataclass(eq=False)
class Island:
  x: float
  y: float
  used: bool = False


@dataclass(eq=False)
class Bridge:
  x: float
  y: float
  target: Island
  origin: Island | None
  length: float


class EchoBridgeEnvironment:
  score_cap = 5000

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
    self.x, self.y, self.angle = 50.0, 0.0, 0.0
    self.material, self.spawn_y, self.last_x = 80.0, -10.0, 50.0
    self.island: Island | None = Island(50, 0)
    self.islands = [self.island]
    self.bridges: list[Bridge] = []
    self.active: Bridge | None = None

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
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _candidates(self):
    for island in self.islands:
      if island is self.island or island.used or not 0 <= island.y <= 100:
        continue
      dx, dy = island.x - self.x, island.y - self.y
      distance = math.hypot(dx, dy)
      if 0 < distance <= 4 * self.material:
        yield island, dx, dy, distance

  def _advance(self, tap: bool) -> int:
    self.angle = (self.angle + 0.05) % math.tau
    scroll = 0.15 if self.island is not None and next(self._candidates(), None) is None else 0.05
    scroll = max(scroll, 60 - self.y)
    self.y += scroll
    self.spawn_y += scroll
    for island in self.islands:
      island.y += scroll
    for bridge in self.bridges:
      bridge.y += scroll
    while self.spawn_y > -10:
      x = self.random.uniform(0, 100)
      for _ in range(9):
        if abs(x - self.last_x) >= 25:
          break
        x = self.random.uniform(0, 100)
      if abs(x - self.last_x) < 25:
        x = self.random.uniform(60, 100) if self.last_x < 50 else self.random.uniform(0, 40)
      self.islands.append(Island(x, self.spawn_y))
      self.last_x = x
      self.spawn_y -= self.random.uniform(5, 20)
    if tap:
      self._build()
    reward = self._walk() if self.pressed and self.active is not None else 0
    self.islands = [island for island in self.islands if island.y <= 120]
    self.bridges = [bridge for bridge in self.bridges if min(bridge.y, bridge.target.y) <= 120]
    if self.active not in self.bridges:
      self.active = None
    self.terminal = self.y >= 100
    return reward

  def _build(self) -> None:
    cos, sin = math.cos(self.angle), math.sin(self.angle)
    best = 0.3
    chosen = None
    for island, dx, dy, distance in self._candidates():
      alignment = (dx * cos + dy * sin) / distance
      if alignment > best:
        best = alignment
        chosen = Bridge(self.x, self.y, island, self.island, distance)
    if chosen is None:
      self.material = max(0, self.material - 10)
      return
    self.active = chosen
    self.bridges.append(chosen)
    chosen.target.used = True
    self.material -= chosen.length * 0.25

  def _walk(self) -> int:
    bridge = self.active
    dx, dy = bridge.target.x - self.x, bridge.target.y - self.y
    distance = math.hypot(dx, dy)
    if distance:
      self.x += dx / distance * 1.5
      self.y += dy / distance * 1.5
    self.island = None
    multiplier = len(self.bridges)
    reward = math.floor(1.5 * multiplier)
    if distance < 9.5:
      self.island = bridge.target
      self.x, self.y = self.island.x, self.island.y
      recovery = bridge.length * 0.2 * multiplier
      self.material = min(80, self.material + recovery)
      if bridge.origin in self.islands:
        self.islands.remove(bridge.origin)
      self.bridges.remove(bridge)
      self.active = None
      reward += math.floor(recovery)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)

    def mark(x: float, y: float, channel: int) -> None:
      if 0 <= y < 100 and 0 <= x <= 100:
        state[int(y) // 10, min(9, int(x) // 10), channel] = True

    mark(self.x, self.y, 0)
    for island in self.islands:
      mark(island.x, island.y, 2 if island.used else 1)
    for bridge in self.bridges:
      for fraction in np.linspace(0, 1, 16):
        mark(bridge.x + fraction * (bridge.target.x - bridge.x),
             bridge.y + fraction * (bridge.target.y - bridge.y), 3)
    mark(self.x + 15 * math.cos(self.angle), self.y + 15 * math.sin(self.angle), 4)
    for row, value in enumerate((self.steps, round(self.material), round(self.angle * 80), len(self.bridges))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.active is not None
    return state
