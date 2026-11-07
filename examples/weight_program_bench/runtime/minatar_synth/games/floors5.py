from dataclasses import dataclass
import math

import numpy as np


CAR_X = 10


@dataclass(eq=False)
class Floor:
  x: float
  y: float
  width: float
  color: int
  paint_from: float = 0
  paint_to: float = 0


class Floors5Environment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.jumps = self.grace = 0
    self.pressed = self.terminal = False
    self.y, self.vy, self.speed = 10.0, 0.0, 1.0
    self.support: Floor | None = None
    self.visited: set[int] = set()
    self.multiplier = 1
    self.floors = [Floor(x, y, width, color) for color, (x, y, width) in enumerate(
      ((25, 30, 10), (52, 50, 35), (105, 70, 30), (160, 60, 30), (220, 40, 50))
    )]

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
    for floor in self.floors:
      if floor.x + floor.width < 0:
        floor.x, floor.y = self.random.uniform(200, 250), self.random.uniform(30, 90)
        floor.width = self.random.uniform(20, 60)
        floor.paint_from = floor.paint_to = 0
      floor.x -= self.speed
    self.speed += 0.02
    self.grace = max(0, self.grace - 1)
    reward = 0
    if self.support is not None:
      floor = self.support
      floor.paint_to = min(floor.width, max(0, CAR_X + 5 - floor.x))
      if tap or floor.x + floor.width < CAR_X - 3:
        width = floor.paint_to - floor.paint_from
        reward = min(999, max(0, math.floor(width) * self.multiplier))
        if width >= floor.width:
          reward *= 3
        floor.x = -999
        self.support = None
        self.speed = 1
        if tap:
          self.vy = -2
          self.jumps += 1
        else:
          self.jumps = 0
          self.grace = 9
    else:
      old_y = self.y
      self.vy += 0.03 if self.pressed else 0.18
      self.y += self.vy
      for floor in self.floors:
        if not floor.x - 3 <= CAR_X <= floor.x + floor.width + 3:
          continue
        if self.vy >= 0 and old_y <= floor.y - 3 <= self.y:
          self.y, self.vy, self.speed = floor.y - 3, 0, 1
          self.support = floor
          floor.paint_from = max(0, CAR_X - 5 - floor.x)
          floor.paint_to = min(floor.width, max(0, CAR_X + 5 - floor.x))
          self.jumps = 0
          self.visited.add(floor.color)
          if len(self.visited) == 5:
            self.multiplier += 1
            self.visited.clear()
          break
        if self.vy < 0 and self.y <= floor.y + 9 <= old_y:
          self.y = floor.y + 9
          self.vy *= -0.7
          break
      if tap and self.support is None and (self.grace or self.jumps < 2):
        self.vy, self.speed = -2, 1
        self.jumps += 1
    self.terminal = self.y > 99
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), 0, 0] = True
    for floor in self.floors:
      row = int(floor.y) // 10
      for col in range(10):
        if col * 20 < floor.x + floor.width and (col + 1) * 20 > floor.x:
          for bit in range(3):
            state[row, col, bit + 1] |= bool((floor.color + 1) & (1 << bit))
          state[row, col, 4] |= floor.paint_to > floor.paint_from and (
            col * 20 < floor.x + floor.paint_to and (col + 1) * 20 > floor.x + floor.paint_from
          )
    for row, value in enumerate((self.steps, self.multiplier, self.jumps, self.grace,
                                 round(self.vy * 16) + 128)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[8, :5, 5] = [color in self.visited for color in range(5)]
    state[9, :2, 5] = self.pressed, self.support is not None
    return state
