from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Mover:
  x: float
  y: float
  vx: float
  vy: float


class GhostHopEnvironment:
  score_cap = 4000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.targets = [
      Mover(20 + i % 3 * 30, 25 + i // 3 * 40, *self.random.uniform(-0.2, 0.2, 2))
      for i in range(5)
    ]
    self.index = 2
    self.x, self.y = 80, 25
    self.origin = (self.x, self.y)
    self.flight = self.steps = self.frames = 0
    self.multiplier = 1.0
    self.pressed = self.terminal = False
    self.bullets: list[Mover] = []

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

  def _advance(self, tap: bool) -> int:
    current = self.targets[self.index]
    if self.frames % 50 == 0:
      offset = self.random.uniform(0, 100)
      side = self.random.integers(4)
      x, y = ((-5, offset), (105, offset), (offset, -5), (offset, 105))[side]
      angle = math.atan2(current.y - y, current.x - x)
      self.bullets.append(Mover(x, y, 0.5 * math.cos(angle), 0.5 * math.sin(angle)))
    self.frames += 1
    if tap and not self.flight:
      self.origin = (current.x, current.y)
      self.flight = 13
    for target in self.targets:
      target.x += target.vx
      target.y += target.vy
      if not 15 <= target.x <= 85:
        target.x = min(85, max(15, target.x))
        target.vx *= -1
      if not 15 <= target.y <= 85:
        target.y = min(85, max(15, target.y))
        target.vy *= -1
    reward = 0
    airborne = self.flight > 0
    if airborne:
      destination = (self.index + 1) % len(self.targets)
      target = self.targets[destination]
      self.flight -= 1
      fraction = (13 - self.flight) * 0.08
      self.x = self.origin[0] + (target.x - self.origin[0]) * fraction
      self.y = self.origin[1] + (target.y - self.origin[1]) * fraction
      if not self.flight:
        self.index = destination
        reward = int(self.multiplier)
        self.multiplier *= 2
    else:
      self.x, self.y = current.x, current.y
    for bullet in self.bullets:
      bullet.x += bullet.vx
      bullet.y += bullet.vy
      if airborne and abs(bullet.x - self.x) < 4 and abs(bullet.y - self.y) < 4:
        self.terminal = True
    self.bullets = [b for b in self.bullets if -10 <= b.x <= 110 and -10 <= b.y <= 110]
    self.multiplier = min(256, max(1, self.multiplier * 0.99))
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y) // 10, int(self.x) // 10, 0] = True
    destination = (self.index + 1) % len(self.targets)
    for index, target in enumerate(self.targets):
      state[int(target.y) // 10, int(target.x) // 10, 2 if index == destination else 1] = True
    for bullet in self.bullets:
      if 0 <= bullet.x < 100 and 0 <= bullet.y < 100:
        row, col = int(bullet.y) // 10, int(bullet.x) // 10
        state[row, col, 3] = True
        state[min(9, max(0, int(bullet.y + 10 * bullet.vy) // 10)),
              min(9, max(0, int(bullet.x + 10 * bullet.vx) // 10)), 4] = True
    for row, value in enumerate((self.steps, self.flight, int(self.multiplier))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
