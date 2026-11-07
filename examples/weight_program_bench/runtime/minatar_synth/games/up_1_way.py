from dataclasses import dataclass, field

import numpy as np


@dataclass
class Floor:
  holes: list[float] = field(default_factory=list)
  timer: float = float("inf")
  bamboo: list[float] = field(default_factory=list)
  skulls: list[float] = field(default_factory=list)
  powerups: list[float] = field(default_factory=list)


def _height(index):
  return 16 + 15 * index


class Up1WayEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.floors = [Floor([self.random.uniform(99, 180)], self.random.uniform(0, 99))
                   for _ in range(5)] + [Floor()]
    self.floor = 5
    self.target = None
    self.y = _height(5)
    self.bamboo_timer = 0.
    self.bamboo_floor = int(self.random.integers(6))
    self.skull_timer = self.random.uniform(49, 99)
    self.power_timer = 999
    self.action = self.steps = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    jump = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(jump and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, jump):
    if self.target is not None:
      height = _height(self.target)
      if abs(height - self.y) < 3:
        self.floor, self.target = self.target, None
        self.y = height
      else:
        self.y += 3 if height > self.y else -3
    if self.target is None:
      if jump and self.floor > 0:
        self.target = self.floor - 1
      elif any(14 < hole < 17 for hole in self.floors[self.floor].holes):
        self.target = self.floor + 1
    self._spawn_items()
    reward = 0
    for index, floor in enumerate(self.floors):
      floor.timer -= 1
      if floor.timer < 0:
        floor.holes.append(200)
        floor.timer = self.random.uniform(32, 99)
      floor.holes = [x - 1 for x in floor.holes if x >= -8]
      near_player = abs(self.y - _height(index)) < 5
      if near_player:
        reward += sum(abs(x - 20) < 5 for x in floor.bamboo)
        floor.bamboo = [x for x in floor.bamboo if abs(x - 20) >= 5]
        if any(abs(x - 20) < 5 for x in floor.powerups):
          floor.powerups = [x for x in floor.powerups if abs(x - 20) >= 5]
          for level in self.floors:
            level.bamboo.extend(level.skulls)
            level.skulls.clear()
        self.terminal |= any(abs(x - 20) < 5 for x in floor.skulls)
      floor.bamboo = [x - 1 for x in floor.bamboo if x >= -2]
      floor.powerups = [x - 1 for x in floor.powerups if x >= -2]
      floor.skulls = [x - 1 for x in floor.skulls if x >= -2 and
                      all(abs(x - item - 1) >= 5 for item in floor.bamboo + floor.powerups)]
    return reward

  def _spawn_items(self):
    self.bamboo_timer -= 1
    if self.bamboo_timer < 0:
      floor = self.floors[self.bamboo_floor]
      if floor.timer < 9:
        self.bamboo_floor = int(self.random.integers(6))
      else:
        floor.bamboo.append(209)
        if self.random.random() < 0.3:
          self.bamboo_timer = 6
        else:
          self.bamboo_timer = self.random.uniform(200, 300)
          self.bamboo_floor = int(self.random.integers(6))
    self.skull_timer -= 1
    if self.skull_timer < 0:
      floor = self.floors[int(self.random.integers(6))]
      if floor.timer > 9 and self.bamboo_timer > 9:
        floor.skulls.append(209)
      self.skull_timer += self.random.uniform(30, 50)
    self.power_timer -= 1
    if self.power_timer < 0:
      self.floors[int(self.random.integers(6))].powerups.append(209)
      self.power_timer = 999

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[(self.y - 5) // 10, 1, 0] = True
    for index, floor in enumerate(self.floors):
      row = (_height(index) - 5) // 10
      for channel, items in enumerate((floor.holes, floor.bamboo, floor.skulls, floor.powerups), 1):
        for x in items:
          if 0 <= x < 200:
            state[row, int(x // 20), channel] = True
    values = (self.steps, self.action, self.floor, 0 if self.target is None else self.target + 1,
              self.y, round(self.bamboo_timer), round(self.skull_timer), self.power_timer)
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
