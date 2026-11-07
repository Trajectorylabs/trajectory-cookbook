from collections import deque
import math

import numpy as np


EARTH = 3


class GeocentEnvironment:
  score_cap = 100

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
    self.phases = self.random.uniform(0, math.tau, 9)
    self.planets = np.zeros((9, 2))
    self.crates: list[int] = []
    self.history: deque[tuple[float, float]] = deque()
    self.turn = 0.0
    self._launch()
    self._locate_planets()

  def _launch(self) -> None:
    self.angle = (self.random.integers(4) + self.random.uniform(0.3, 0.7)) * math.pi / 2
    self.distance = abs(math.sin(2 * self.angle)) * 25 + 36
    self.speed = 0.0
    self.crates = list(range(9, 9 * (len(self.crates) + 1) + 1, 9))
    self.history = deque(maxlen=max(self.crates) + 1)

  def _locate_planets(self) -> None:
    radii = np.arange(9) * 4
    self.planets[:, 0] = radii * np.cos(self.phases)
    self.planets[:, 1] = radii * np.sin(self.phases)
    self.planets += 50 - self.planets[EARTH].copy()

  @property
  def position(self) -> tuple[float, float]:
    return 50 + self.distance * math.cos(self.angle), 50 + self.distance * math.sin(self.angle)

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

  def _hazard(self, x: float, y: float, margin: float) -> bool:
    return any(math.hypot(px - x, py - y) < margin + (3 if index == 0 else 1)
               for index, (px, py) in enumerate(self.planets) if index != EARTH)

  def _advance(self, tap: bool) -> int:
    self.phases += 0.03 * 0.7 ** np.arange(1, 10)
    self._locate_planets()
    if tap:
      self.speed += 1
    self.speed += ((0.5 if self.pressed else 0.1) - self.speed) * 0.5
    self.distance -= self.speed
    self.turn += ((0.03 if self.pressed else -0.003) - self.turn) * 0.05
    self.angle = (self.angle + self.turn) % math.tau
    x, y = self.position
    self.history.appendleft((x, y))
    self.crates = [age for age in self.crates
                   if not self._hazard(*(self.history[age] if age < len(self.history) else (x, y)), 1)]
    if self.distance < 3.5:
      reward = len(self.crates)
      self._launch()
      return reward
    self.terminal = self._hazard(x, y, 1.5)
    return 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)

    def mark(x: float, y: float, channel: int) -> None:
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, channel] = True

    for index, (x, y) in enumerate(self.planets):
      mark(x, y, 2 if index == EARTH else 1)
    x, y = self.position
    mark(min(99, max(0, x)), min(99, max(0, y)), 0)
    for age in self.crates:
      mark(*(self.history[age] if age < len(self.history) else (x, y)), 3)
    mark(50 + (self.distance - 8) * math.cos(self.angle),
         50 + (self.distance - 8) * math.sin(self.angle), 4)
    for row, value in enumerate((self.steps, len(self.crates), round(self.distance * 4), round(self.angle * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
