from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


HORIZON = 400
UNIT = 8
WATER = 72
PERIOD = 32


@dataclass
class Bar:
  anchor: int
  length: int
  phase: int
  used: bool = False

  def tip(self, ceiling: int) -> tuple[int, int]:
    angle = 0.85 * sin(self.phase * 2 * pi / PERIOD)
    return self.anchor + round(self.length * sin(angle)), ceiling + round(self.length * cos(angle))


class AerialBarEnvironment:
  score_cap = HORIZON

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.flight = 0
    self.ceiling = 8
    self.attached: Bar | None = Bar(32, 38, 0, True)
    self.bars = [self.attached, Bar(66, int(self.random.integers(25, 34)), 0)]
    self.x, self.y = self.attached.tip(self.ceiling)
    self.vx = self.vy = 0
    self.bounced = False
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    scroll = int(self.steps % 8 == 0) + max(0, (self.x - 40) // 8)
    self.x -= scroll
    for bar in self.bars:
      bar.anchor -= scroll
      bar.phase = (bar.phase + 1) % PERIOD
    if self.bars[-1].anchor < 55:
      self.bars.append(Bar(self.bars[-1].anchor + int(self.random.integers(26, 42)),
                           int(self.random.integers(18, 39)), int(self.random.integers(PERIOD))))

    reward = 0
    if self.attached is not None:
      bar = self.attached
      self.x, self.y = bar.tip(self.ceiling)
      if pressed:
        future = Bar(bar.anchor, bar.length, (bar.phase + 1) % PERIOD)
        next_x, next_y = future.tip(self.ceiling)
        self.vx, self.vy = next_x - self.x, next_y - self.y - 1
        self.attached = None
        self.flight = 0
        self.bounced = False
    else:
      self.flight += 1
      self.x += self.vx
      self.y += self.vy
      if action != 5 or self.steps % 5 == 0:
        self.vy += 1
      if action != 5 and self.steps % 4 == 0:
        self.vx -= int(np.sign(self.vx))
      if self.y >= WATER:
        self.y = WATER - 1
        self.vy = -max(1, self.vy * 3 // 2)
        self.ceiling += 8
        self.bounced = True
        for bar in self.bars:
          bar.used = False
      if self.y <= self.ceiling:
        self.y = self.ceiling
        self.vy = max(0, -self.vy // 2)
      for bar in self.bars:
        x, y = bar.tip(self.ceiling)
        if not bar.used and abs(self.x - x) <= 5 and abs(self.y - y) <= 5:
          self.attached = bar
          bar.used = True
          reward = 0 if self.bounced else self.flight
          self.ceiling = max(8, self.ceiling - 4)
          self.x, self.y = bar.tip(self.ceiling)
          break
    self.bars = [bar for bar in self.bars if bar.anchor >= -24 or bar is self.attached]
    self.terminal = (
      (self.attached.anchor < 0 if self.attached is not None else self.x < 0)
      or any(bar.tip(self.ceiling)[1] >= WATER for bar in self.bars)
      or self.steps >= self.horizon
    )
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, self.y // UNIT), max(0, min(9, self.x // UNIT)), 0] = True
    state[WATER // UNIT:, :, 1] = True
    for bar in self.bars:
      x, y = bar.tip(self.ceiling)
      if 0 <= x < 80 and 0 <= y < 80:
        state[y // UNIT, x // UNIT, 3 if bar.used else 2] = True
    state[:min(10, self.ceiling // UNIT), :, 4] = True
    values = (self.steps, self.flight, self.ceiling, self.x + 32, self.y,
              self.vx + 16, self.vy + 64, int(self.attached is not None),
              int(self.bounced), int(self.last_action == 5))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
