from collections import deque
from math import ceil, sqrt

import numpy as np


_PATTERNS = ((0, 0, 0), (0, 1, 0), (0, 3, 0), (0, 0, 1), (0, 0, 3), (1, 0, 0), (3, 0, 0))
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_BOARD = tuple((row, col) for row in range(10) for col in range(10))
_BORDER = tuple(cell for cell in _BOARD if 0 in cell or 9 in cell)


def _neighbors(cell):
  row, col = cell
  for dr, dc in _DIRECTIONS:
    neighbor = row + dr, col + dc
    if 0 <= neighbor[0] < 10 and 0 <= neighbor[1] < 10:
      yield neighbor


class TtFenceEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.tip = (4, 4)
    initial = self._path(int(self.random.integers(7)), int(self.random.integers(4)))
    self.fence = set(initial)
    self.tip = initial[-1]
    self.pattern = int(self.random.integers(7))
    self.angle = 0
    self.rotation_timer = 0
    self.blast = set()
    self.blast_timer = 0
    self.damage = self.damage_target = 0.
    self.steps = self.action = 0
    self.terminal = False
    self.preview = self._rotate()
    self.rotation_timer = 30

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    place = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(place and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, place):
    self.blast_timer -= 1
    reward = 0
    if place and self.preview:
      self.damage_target += 2 + 9 * sum(cell in self.fence for cell in self.preview)
      self.fence.update(self.preview)
      self.tip = self.preview[-1]
      reward = self._bomb()
      self.pattern = int(self.random.integers(7))
      self.rotation_timer = 0
    self.rotation_timer -= 1
    if self.rotation_timer < 0:
      self.preview = self._rotate()
      self.rotation_timer = 30
    self.damage_target += 0.01
    self.damage += (self.damage_target - self.damage) * 0.1
    self.terminal = self.damage > 99
    return reward

  def _path(self, pattern, angle):
    row, col = self.tip
    cells = []
    for offset in (0, *_PATTERNS[pattern]):
      dr, dc = _DIRECTIONS[(angle + offset) % 4]
      row, col = row + dr, col + dc
      cells.append((row, col))
    return cells

  def _rotate(self):
    fallback = None
    for turn in range(1, 5):
      angle = (self.angle + turn) % 4
      path = self._path(self.pattern, angle)
      if all(0 <= row < 10 and 0 <= col < 10 for row, col in path):
        if not self.fence.intersection(path):
          self.angle = angle
          return path
        if fallback is None:
          fallback = angle, path
    if fallback is not None:
      self.angle, path = fallback
      return path
    self.damage_target = 100
    return []

  def _bomb(self):
    outside = set(_BORDER) - self.fence
    queue = deque(outside)
    while queue:
      for cell in _neighbors(queue.popleft()):
        if cell not in outside and cell not in self.fence:
          outside.add(cell)
          queue.append(cell)
    enclosed = set(_BOARD) - outside - self.fence
    self.blast = {neighbor for cell in enclosed for neighbor in _neighbors(cell)}
    for _ in range(int(sqrt(len(self.blast)) * 0.5)):
      for cell in _BOARD:
        if cell in self.blast:
          self.blast.update(neighbor for neighbor in _neighbors(cell) if neighbor in self.fence)
    self.blast_timer = 30
    self.fence.difference_update(self.blast)
    count = len(self.blast)
    reward = ceil(count * sqrt(count))
    if reward:
      self.damage_target = min(99, max(0, self.damage_target - reward * 0.1))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for row, col in self.fence:
      state[row, col, 0] = True
    for row, col in self.preview:
      state[row, col, 1] = True
    for row, col in self.blast if self.blast_timer > 0 else ():
      state[row, col, 2] = True
    state[self.tip[0], self.tip[1], 3] = True
    state[9, :min(10, ceil(self.damage / 10)), 4] = True
    values = (self.steps, self.action, self.pattern, self.angle, self.rotation_timer,
              round(self.damage * 4), round(self.damage_target * 4))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
