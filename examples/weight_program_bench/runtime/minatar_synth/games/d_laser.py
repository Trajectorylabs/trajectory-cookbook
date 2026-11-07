from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Shield:
  x: float
  y: float
  target: float


@dataclass
class Laser:
  bottom: float = 20
  blocked: bool = False


class DLaserEnvironment:
  score_cap = 500

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
    self.x, self.y, self.direction = 50.0, 90.0, 1
    self.camera = 80.0
    self.distance = self.charge = self.charge_speed = self.acceleration = 0.0
    self.shield_distance = 0.0
    self.shields: list[Shield] = []
    self.lasers: list[Laser] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction *= -1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    scroll = 0.05 + max(0, self.camera - self.y) * 0.1
    self.camera += (80 - self.camera) * 0.001
    self.shield_distance -= scroll
    if self.shield_distance < 0:
      target = 5 + 10 * int(self.random.integers(10))
      self.shields.append(Shield(0 if target < 50 else 99, 15, target))
      self.shield_distance += self.random.uniform(9, 36)
    for shield in self.shields:
      shield.y += scroll
      delta = shield.target - shield.x
      shield.x = shield.target if abs(delta) < 1 else shield.x + delta * 0.2
    self.shields = [shield for shield in self.shields if shield.y <= 99]

    reward = 0
    if self.charge > 10:
      self.charge += 9
      if not self.lasers:
        self.lasers = [Laser() for _ in range(10)]
      if self.charge > 199:
        reward = math.ceil(self.distance**1.5 * 0.1)
        self._reset_lasers()
    else:
      self.acceleration += self.random.uniform(-0.0001, 0.0001)
      self.charge_speed += self.acceleration
      if self.charge_speed < 0 and self.acceleration < 0 or self.charge_speed > 0.2 and self.acceleration > 0:
        self.acceleration *= -1
        self.charge_speed *= self.random.uniform(0.2, 1.5)
      self.charge += self.charge_speed

    for column, laser in enumerate(self.lasers):
      if laser.blocked:
        laser.bottom += scroll
      else:
        laser.bottom += 9
        stops = [shield.y - 2 for shield in self.shields if shield.x == 5 + 10 * column
                 and 10 <= shield.y - 2 <= laser.bottom]
        if stops:
          laser.bottom = min(stops)
          laser.blocked = True

    self.y += scroll
    if not self.pressed:
      self.x += self.direction
      if not 0 <= self.x <= 99:
        self.x = min(99, max(0, self.x))
        self.direction *= -1
      self.y -= 0.5
      self.distance += 0.5
    if self.lasers and any(
      10 * column - 2 <= self.x <= 10 * column + 12 and 8 <= self.y <= laser.bottom + 2
      for column, laser in enumerate(self.lasers)
    ):
      self.camera += 7
      self.y += 7
      self._reset_lasers()
    self.terminal = self.y > 99
    return reward

  def _reset_lasers(self) -> None:
    self.lasers.clear()
    self.distance = self.charge = 0.0
    self.charge_speed *= self.random.random()
    self.acceleration *= self.random.random()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = min(9, max(0, int(self.y) // 10)), int(self.x) // 10
    state[row, col, 0] = True
    state[row, col, 4] = self.direction > 0
    for shield in self.shields:
      state[int(shield.y) // 10, int(shield.x) // 10, 1 if shield.x == shield.target else 2] = True
    for col, laser in enumerate(self.lasers):
      state[1:min(10, int(laser.bottom) // 10 + 1), col, 3] = True
    for row, value in enumerate((self.steps, round(max(0, self.charge)), round(self.distance))):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
