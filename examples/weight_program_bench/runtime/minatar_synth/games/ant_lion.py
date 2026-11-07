from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


HORIZON = 400
ANGLES = 32


@dataclass
class EscapeBar:
  angle: int
  inner: int
  outer: int
  speed: int = 2


class AntLionEnvironment:
  score_cap = HORIZON * (HORIZON + 1)

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.angle, self.radius = 8, 32
    self.steps = self.last_action = self.velocity = self.multiplier = 0
    self.spawn_in = 1
    self.bars: list[EscapeBar] = []
    self.attached: EscapeBar | None = None
    self.offset = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    if self.attached is None:
      self.velocity = max(-2, min(2, self.velocity + (1 if action == 5 else -1)))
      self.angle = (self.angle + self.velocity) % ANGLES
      self.radius -= 2 if self.radius > 40 else 1
    if self.radius <= 3:
      self.terminal = True
      return 0.0, True
    self.spawn_in -= 1
    if not self.spawn_in:
      self.bars.append(EscapeBar(int(self.random.integers(ANGLES)), 42,
                                 42 + int(self.random.integers(8, 17))))
      self.spawn_in = max(8, int(self.random.integers(14, 20)) - self.steps // 80)
    remaining = []
    for bar in self.bars:
      if self.steps % 4 == 0:
        bar.angle = (bar.angle - 1) % ANGLES
      movement = bar.speed + int(bar.inner > 40)
      bar.inner = max(0, bar.inner - movement)
      bar.outer -= movement
      if bar.outer >= 0:
        remaining.append(bar)
      else:
        self.multiplier = max(1, self.multiplier - 1) if self.multiplier else 0
        if self.attached is bar:
          self.attached = None
    self.bars = remaining
    if self.attached is not None:
      bar = self.attached
      self.offset += 3
      if self.steps % 5 == 0:
        bar.speed = min(3, bar.speed + 1)
      self.angle, self.radius = bar.angle, bar.inner + self.offset
      if self.radius > bar.outer + 3:
        self.attached = None
    reward = 0
    if self.attached is None and self.radius > 3:
      for bar in self.bars:
        separation = (self.angle - bar.angle) % ANGLES
        if min(separation, ANGLES - separation) <= 1 and bar.inner <= self.radius <= bar.outer:
          self.attached = bar
          self.offset = self.radius - bar.inner
          self.multiplier += 2
          reward = self.multiplier
          break
    self.terminal = self.radius <= 3 or self.steps >= self.horizon
    return float(reward), self.terminal

  @staticmethod
  def _cell(angle: int, radius: int) -> tuple[int, int]:
    radians = angle * 2 * pi / ANGLES
    return round(4.5 + sin(radians) * radius / 9), round(4.5 + cos(radians) * radius / 9)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, column = self._cell(self.angle, self.radius)
    state[max(0, min(9, row)), max(0, min(9, column)), 0] = True
    state[4:6, 4:6, 1] = True
    for bar in self.bars:
      for radius in range(bar.inner, bar.outer + 1):
        row, column = self._cell(bar.angle, radius)
        if 0 <= row < 10 and 0 <= column < 10:
          state[row, column, 3 if bar is self.attached else 2] = True
    values = (self.angle, self.radius, self.velocity + 2, self.offset,
              self.multiplier, self.steps, self.spawn_in, self.last_action)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
