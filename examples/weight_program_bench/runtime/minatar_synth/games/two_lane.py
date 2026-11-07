from dataclasses import dataclass
from math import floor, sqrt

import numpy as np


@dataclass
class Road:
  y: float
  lanes: tuple[int, ...] = (30, 70)
  coin: int | None = None


class TwoLaneEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.roads = [Road(y) for y in range(-10, 110, 10)]
    self.lanes = (30, 70)
    self.speed = 1.
    self.x = 30.
    self.target = 30
    self.multiplier = 1
    self.off_road = self.terminal = False
    self.steps = self.action = 0

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    if self.action == 5 and previous != 5:
      self.target = 100 - self.target
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.off_road:
      self.speed *= 0.9
      if self.speed < 0.1:
        self.terminal = True
        return 0
    elif self.action == 5:
      self.speed = min(3, self.speed + 0.02)
    else:
      self.speed += (1 - self.speed) * 0.1
    self.x += (self.target - self.x) * 0.15 * sqrt(self.speed)
    reward = 0
    on_road = False
    for road in self.roads:
      road.y += self.speed
      if road.y >= 110:
        road.y -= 120
        self._renew(road)
      if road.coin is not None and abs(self.x - road.coin) < 5 and abs(85 - road.y) < 5:
        reward += floor(self.speed ** 2 + 1) * self.multiplier
        self.multiplier = min(32, self.multiplier + 1)
        road.coin = None
      if road.y < 93 and road.y + 10 > 87:
        on_road |= any(abs(self.x - lane) < 23 for lane in road.lanes)
    self.off_road = not on_road
    if self.off_road:
      self.multiplier = 1
    return reward

  def _renew(self, road):
    if self.random.random() < 0.1:
      self.lanes = ((int(self.random.choice((30, 70))),)
                    if len(self.lanes) == 2 else (30, 70))
    road.lanes = self.lanes
    if road.coin is not None:
      self.multiplier = max(1, self.multiplier - 1)
    road.coin = int(self.random.choice(road.lanes)) if self.random.random() < 0.2 else None

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x // 10), 0] = True
    for road in self.roads:
      for row in range(max(0, floor(road.y / 10)), min(10, floor((road.y + 9.999) / 10) + 1)):
        for lane in road.lanes:
          col = lane // 10
          state[row, col - 2:col + 2, 1] = True
      if road.coin is not None and 0 <= road.y + 5 < 100:
        state[int((road.y + 5) // 10), road.coin // 10, 2] = True
    state[9, self.target // 10, 3] = True
    state[9, int(self.x // 10), 4] = self.off_road
    values = (self.steps, self.action, round(self.speed * 100), self.multiplier,
              round(self.x * 5), len(self.lanes))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
