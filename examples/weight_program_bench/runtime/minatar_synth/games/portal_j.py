from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class Floor:
  x: float
  width: float


class PortalJEnvironment:
  score_cap = 3000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.portal = 30.0, 39.0, 36.0
    self.steps = self.phase = self.pending_score = 0
    self.fall = self.floor_distance = 0.0
    self.gold_distance = 30.0
    self.multiplier = 1
    self.floors = [Floor(0, 100), Floor(110, 80)]
    self.gold = []
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
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

  def _advance(self, tap):
    scroll = max(0, self.x - 30) * 0.1 + 0.1
    self.floor_distance -= scroll
    if self.floor_distance < 0:
      interval = self.random.uniform(50, 99)
      fraction = self.random.uniform(0.4, 0.7)
      x = 200 + self.floor_distance + interval * self.random.uniform(0, 1 - fraction)
      self.floors.append(Floor(x, interval * fraction))
      self.floor_distance += interval
    for floor in self.floors:
      floor.x -= scroll
    self.floors = [floor for floor in self.floors if floor.x + floor.width >= 0]
    self.x -= scroll
    self.portal -= scroll
    if self.fall:
      self.fall += 0.2
    self.y = 39 + self.fall
    reward = 0
    if self.phase:
      self.phase += 1
      if self.phase < 5:
        self.y += self.phase
      else:
        distance = self.portal - self.x
        if distance > 0:
          self.pending_score = ceil(distance ** 1.5 * 0.1 * self.multiplier)
          self.x = self.portal
        self.y = 47 - self.phase
        if self.y < 39:
          self.y = 39
          self.phase = 0
          reward = self.pending_score
          self.pending_score = 0
    else:
      self.portal += 2
      if self.portal > 200:
        self.portal = self.x + 6
      if tap:
        self.phase = 1
    if self.phase == 0:
      on_floor = self.y + 2.5 > 40 and self.y - 2.5 < 48 and any(
        self.x + 2.5 > floor.x and self.x - 2.5 < floor.x + floor.width
        for floor in self.floors
      )
      self.fall = 0 if on_floor else self.fall or 1
    self.terminal = self.y > 52 or self.x < 0

    self.gold_distance -= scroll
    if self.gold_distance < 0:
      self.gold.append(203)
      self.gold_distance += self.random.uniform(99, 120)
    remaining = []
    for x in self.gold:
      x -= scroll
      if abs(x - self.x) < 5 and abs(self.y - 36) < 5:
        self.multiplier += 1
      elif x >= -3:
        remaining.append(x)
    self.gold = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    if 0 <= self.x < 200 and 0 <= self.y < 50:
      state[int(self.y // 5), int(self.x // 20), 0] = True
    if 0 <= self.portal < 200:
      state[8, int(self.portal // 20), 1] = True
    columns = np.arange(10) * 20
    for floor in self.floors:
      state[8:10, :, 2] |= (columns < floor.x + floor.width) & (columns + 20 > floor.x)
    for x in self.gold:
      if 0 <= x < 200:
        state[7, int(x // 20), 3] = True
    if self.phase and 0 <= self.x < 200:
      state[8, int(self.x // 20), 4] = True
    for row, value in enumerate((self.steps, self.phase, self.multiplier, self.pressed)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
