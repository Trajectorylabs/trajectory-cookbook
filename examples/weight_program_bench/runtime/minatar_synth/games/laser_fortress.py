from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Unit:
  x: float
  speed: int
  ally: bool


class LaserFortressEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.units: list[Unit] = []
    self.timer = 0.0
    self.group_remaining = 6
    self.fast_index = -1
    self.ally_group = False
    self.sight = 40.0
    self.beam = 0.0
    self.multiplier = 1
    self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.beam = self.sight
      self.multiplier = 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    beam_x = self.beam
    if self.pressed:
      self.beam += 2
    self.timer -= 1
    if self.timer < 0:
      self._spawn()
    reward = 0
    nearest = 200.0
    remaining = []
    for unit in self.units:
      unit.x -= unit.speed
      if not unit.ally:
        nearest = min(nearest, unit.x)
      if self.pressed and abs(beam_x - unit.x) < 5:
        if unit.ally:
          self.terminal = True
        else:
          reward += self.multiplier
          self.multiplier += 1
      elif unit.x < 0:
        self.terminal |= not unit.ally
      else:
        remaining.append(unit)
    self.units = remaining
    if nearest < 200:
      self.sight += (nearest - 8 - self.sight) * 0.3
    return reward

  def _spawn(self) -> None:
    speed = 2 if self.group_remaining - 1 == self.fast_index else 1
    self.units.append(Unit(163, speed, self.ally_group))
    self.group_remaining -= 1
    if self.group_remaining:
      self.timer = self.random.uniform(5, 8)
    else:
      self.group_remaining = 10 - math.floor(math.sqrt(self.random.uniform(0, 50)))
      self.timer = self.random.uniform(20, 30)
      self.ally_group = not self.ally_group and self.random.random() < 0.3
      self.fast_index = -1 if self.random.random() < 0.5 else int(self.random.integers(2))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[1:5, 0, 0] = True
    for unit in self.units:
      if 0 <= unit.x < 160:
        state[7, int(unit.x) // 16, 2 if unit.ally else 1] = True
    if 0 <= self.sight < 160:
      state[7, int(self.sight) // 16, 3] = True
    if self.pressed:
      for fraction in np.linspace(0, 1, 20):
        x = 20 + (self.beam - 2 - 20) * fraction
        y = 8 + 42 * fraction
        if 0 <= x < 160:
          state[int(y) // 6, int(x) // 16, 4] = True
    values = (self.steps, min(511, self.multiplier), min(511, max(0, round(self.sight * 3))),
              min(511, max(0, round(self.beam * 3))))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
