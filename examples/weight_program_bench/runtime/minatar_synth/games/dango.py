from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Dango:
  x: float
  angle: float
  spin: float
  height: float
  factor: int
  multiply: bool
  caught: bool = False

  @property
  def y(self) -> float:
    return 50 + 30 * math.sin(self.angle)


def expression_score(terms: list[tuple[bool, int]]) -> int:
  total = product = 0
  for multiply, value in terms:
    if multiply:
      product *= value
    else:
      total += product
      product = value
  return total + product


class DangoEnvironment:
  score_cap = 10000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.stroke = 0
    self.sticks = 9
    self.offset = 0.0
    self.pressed = self.terminal = False
    self.terms: list[tuple[bool, int]] = []
    self.dangos: list[Dango] = []
    self._refill()

  def _refill(self) -> None:
    while len(self.dangos) < 7:
      self.dangos.append(Dango(
        -6, self.random.uniform(0, math.tau),
        self.random.uniform(0.02, 0.05) * self.random.choice((-1, 1)),
        self.random.uniform(15, 30), int(self.random.integers(2, 10)),
        self.random.random() < 0.5,
      ))

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
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _stick_x(self) -> float:
    if not self.stroke:
      return 90
    return 80 - 10 * self.stroke if self.stroke < 10 else 80 + (self.stroke - 10) * 10

  def _advance_frame(self, tap: bool) -> int:
    if tap and not self.stroke:
      self.stroke = 1
      self.sticks -= 1
      self.terms.clear()
    reward = 0
    if self.stroke:
      self.stroke += 1
      self.offset *= 0.1
      if self.stroke > 19:
        self.stroke = 0
        reward = expression_score(self.terms)
        self.sticks = min(9, self.sticks + reward // 100)
    elif self.sticks < 0:
      self.terminal = True
      return 0

    self.offset += 0.002
    survivors = []
    for dango in self.dangos:
      if dango.caught:
        if self.stroke > 10:
          dango.x += 20
        if dango.x > 99:
          self.terms.insert(0, (dango.multiply, dango.factor))
          continue
      else:
        dango.x += (80 - 12 * len(survivors) - dango.x) * 0.1
        dango.angle += dango.spin
      if abs(dango.y - 76) < dango.height / 2 + 1 and dango.x + self.offset + 6.5 > self._stick_x():
        if 0 < self.stroke < 10:
          dango.caught = True
        elif not self.stroke:
          self.terminal = True
      survivors.append(dango)
    self.dangos = survivors
    self._refill()
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for index, dango in enumerate(self.dangos):
      x = dango.x + self.offset
      if 0 <= x < 100:
        col = int(x) // 10
        low = max(0, int(dango.y - dango.height / 2) // 10)
        high = int(dango.y + dango.height / 2) // 10
        state[low:high + 1, col, 2 if dango.caught else int(dango.multiply)] = True
      for bit in range(4):
        state[0, index, bit] = bool(dango.factor & (1 << bit))
      state[0, index, 4] = dango.multiply
    if self.sticks >= 0 or self.stroke:
      state[7, max(0, int(self._stick_x()) // 10):, 3] = True
    for row, value in enumerate((self.steps, self.sticks + 1, self.stroke), 1):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
