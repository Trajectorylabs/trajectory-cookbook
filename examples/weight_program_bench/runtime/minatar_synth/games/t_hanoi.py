from math import floor, sqrt

import numpy as np


class THanoiEnvironment:
  score_cap = 1000
  action_rods = {1: 0, 2: 1, 3: 2}

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(400, horizon)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.rods = [[], [], []]
    self.heights = [90.] * 3
    self.selected = None
    self.steps = self.action = self.timer = self.hold = 0
    self.multiplier = 100
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
    rod = self.action_rods.get(self.action)
    tap = rod is not None and previous not in self.action_rods
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(rod, tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, rod, tap):
    if not self.rods[0]:
      self.timer = 0
    self.timer -= 1
    if self.timer < 0:
      self.rods[0].insert(0, int(self.random.integers(1, 5)))
      self.timer = 99 * sqrt(len(self.rods[0]))
    if tap:
      if self.selected is None:
        if self.rods[rod]:
          self.selected = rod
      else:
        source = self.rods[self.selected]
        target = self.rods[rod]
        if rod != self.selected and (rod == 0 or not target or source[-1] <= target[-1]):
          target.append(source.pop())
        self.selected = None
    if rod is None:
      self.hold = 0
    else:
      self.hold += 1
      if rod:
        self.heights[rod] -= self.hold ** 2 / 10000
    reward = 0
    for index, disks in enumerate(self.rods):
      self.heights[index] -= 0.01
      if self.heights[index] >= 6 * len(disks):
        continue
      if index == 0:
        self.terminal = True
      else:
        points = len(disks) ** 2
        reward += points * (self.multiplier // 100)
        self.multiplier += points
        disks.clear()
        self.heights[index] = 90.
        self.heights[0] = min(90, max(0, self.heights[0] + points / 4))
        if self.selected == index:
          self.selected = None
        self.hold = 0
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for rod, disks in enumerate(self.rods):
      for index, disk in enumerate(disks):
        row, col = 9 - index % 10, 2 * rod + index // 10
        for bit in range(3):
          state[row, col, bit] = bool(disk & (1 << bit))
      state[9, 2 * rod, 3] = self.selected == rod
      state[9, 2 * rod, 4] = self.action_rods.get(self.action) == rod
    values = (self.steps, floor(max(0, self.timer)), self.hold, self.multiplier,
              *(round(max(0, height) * 10) for height in self.heights))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
