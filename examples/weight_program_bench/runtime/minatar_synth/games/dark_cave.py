from dataclasses import dataclass
import math

import numpy as np


PLAYER_Y = 90
HALF_ANGLE = math.pi / 6


@dataclass
class Object:
  x: float
  y: float
  coin: bool


class DarkCaveEnvironment:
  score_cap = 2000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.x, self.angle, self.light = 50.0, 0.0, 40.0
    self.spike_distance, self.coin_distance = 0.0, 10.0
    self.terminal = False
    self.objects: list[Object] = []

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
    direction = -1 if action == 1 else 1 if action == 4 else 0
    reward = 0
    for _ in range(4):
      reward += self._advance_frame(direction)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, direction: int) -> int:
    scroll = 0.1 + self.light * 0.01
    self.spike_distance -= scroll
    self.coin_distance -= scroll
    if self.spike_distance < 0:
      self.objects.append(Object(self.random.uniform(5, 95), 0, False))
      self.spike_distance = self.random.uniform(20, 30)
    if self.coin_distance < 0:
      self.objects.append(Object(self.random.uniform(10, 90), 0, True))
      self.coin_distance = self.random.uniform(30, 40)
    next_x = min(97, max(3, self.x + 1.5 * direction))
    tilt = min(0.1, max(-0.1, next_x - self.x))
    self.angle = min(math.pi / 8, max(-math.pi / 8, self.angle + tilt))
    self.x = next_x
    self.light *= 0.999

    reward = 0
    survivors = []
    for obj in self.objects:
      obj.y += scroll
      if abs(obj.x - self.x) < 6 and abs(obj.y - PLAYER_Y) < 6:
        if obj.coin:
          reward += math.floor(self.light)
          self.light = min(99, self.light + 9)
        else:
          self.light /= 3
      elif obj.y <= 100:
        survivors.append(obj)
    self.objects = survivors
    self.terminal = self.light < 9
    return reward

  def _in_cone(self, x: float, y: float) -> bool:
    return abs(math.atan2(x - self.x, PLAYER_Y - y) - self.angle) < HALF_ANGLE

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x) // 10, 0] = True
    for row in range(9):
      for column in range(10):
        x, y = 10 * column + 5, 10 * row + 5
        state[row, column, 3] = self._in_cone(x, y) and math.hypot(x - self.x, y - PLAYER_Y) < self.light
    for obj in self.objects:
      if obj.y >= 100 or not self._in_cone(obj.x, obj.y):
        continue
      row, column = int(obj.y) // 10, int(obj.x) // 10
      if math.hypot(obj.x - self.x, obj.y - PLAYER_Y) < self.light:
        state[row, column, 1 if obj.coin else 2] = True
      else:
        state[row, column, 4] = True
    for row, value in enumerate((self.steps, round(self.light * 4), round((self.angle + 0.4) * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
