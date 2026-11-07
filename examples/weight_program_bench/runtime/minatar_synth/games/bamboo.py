from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
MIN_SPAWN_INTERVAL = 70
MAX_HARVEST = 100
TRACK_WIDTH = 200


@dataclass
class Stalk:
  x: int
  height: float = 0
  speed: float = 1


class BambooEnvironment:
  score_cap = math.ceil(HORIZON * FRAMES_PER_STEP / MIN_SPAWN_INTERVAL) * MAX_HARVEST

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.pressed = self.terminal = False
    self.x = 190.0
    self.direction = 1
    self.boost = 0.0
    self.stalks: list[Stalk] = []

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
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    self.x = (self.x + self.direction * (1 + self.boost)) % TRACK_WIDTH
    self.boost *= 0.9
    if self.spawn_timer == 0:
      self.stalks.append(Stalk(int(self.random.integers(5, 195))))
      self.spawn_timer = int(self.random.integers(MIN_SPAWN_INTERVAL, 101))
    self.spawn_timer -= 1

    reward = 0
    survivors = []
    for stalk in self.stalks:
      stalk.height += 0.14 * stalk.speed
      distance = abs(self.x - stalk.x)
      if not self.pressed and min(distance, TRACK_WIDTH - distance) <= 6:
        if 5 <= stalk.height <= 25:
          grade = math.ceil((stalk.height - 5) / 3)
          reward += MAX_HARVEST if grade == 7 else grade**2
          continue
        if stalk.height > 25:
          stalk.height *= 0.7
          stalk.speed *= 0.6
          self.direction *= -1
          self.boost = min(5, self.boost + 1)
          if stalk.height <= 25:
            continue
      if stalk.height > 50:
        stalk.speed *= 0.997
      if stalk.height >= 89:
        self.terminal = True
      survivors.append(stalk)
    self.stalks = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    column = int(self.x) // 20
    state[8, column, 0] = True
    state[8, column, 1] = self.direction < 0
    for stalk in self.stalks:
      column = stalk.x // 20
      height = max(1, math.ceil(stalk.height / 10))
      state[9 - height:9, column, 2] = True
      state[9 - height, column, 3] |= 5 <= stalk.height <= 25
      state[9 - height, column, 4] |= 23 < stalk.height <= 25
    state[9, 0, 5] = self.pressed
    for bit in range(9):
      state[9, bit + 1, 5] = bool(self.steps & (1 << bit))
    return state
