from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np


@dataclass
class Item:
  x: float
  y: float
  red: bool
  vx: float = -0.1


class SnakyEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angles = [0., 0.]
    self.velocities = [0., 0.]
    self.items = []
    self.timer = self.steps = 0
    self.red_count = 2
    self.next_y = 50.
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _segments(self):
    length = 50 / len(self.angles) + 3
    x, y = 0., 50.
    for angle in self.angles:
      end_x, end_y = x + length * cos(angle), y + length * sin(angle)
      yield x, y, end_x, end_y
      x, y = end_x, end_y

  def _advance(self):
    limit = pi / (sqrt(len(self.angles) - 1) + 3)
    for index, angle in enumerate(self.angles):
      velocity = self.velocities[index]
      if index == 0:
        updated = min(limit, max(-limit, angle + (-0.03 if self.held else 0.03)))
        velocity = updated - angle
      else:
        updated = angle + velocity
        difference = updated - self.angles[index - 1]
        if abs(difference) > limit and difference * velocity > 0:
          velocity *= -0.2
      self.angles[index] = updated
      self.velocities[index] = velocity * 0.95
      if index + 1 < len(self.angles):
        self.velocities[index + 1] += velocity * 0.95 * 0.07
      if index:
        self.velocities[index - 1] += velocity * 0.95 * 0.01
    segments = list(self._segments())
    self.timer -= 1
    if self.timer < 0:
      self.red_count -= 1
      red = self.red_count < 0
      if red:
        self.red_count = 7
      self.items.append(Item(103, self.next_y, red))
      y = self.random.uniform(20, 80)
      self.timer = 4 * abs(self.next_y - y + 10)
      self.next_y = y
    reward = 0
    remaining = []
    for item in self.items:
      item.vx *= 1.005
      item.x += item.vx
      size = len(self.angles)
      if self._touches(item, segments[-1]):
        reward += size ** 2 if item.red else size
        if item.red and size < 10:
          self.angles.append(0.)
          self.velocities.append(0.)
      elif any(self._touches(item, segment) for segment in segments[:-1]):
        reward += -size if item.red else 1
        if item.red and size > 1:
          self.angles.pop()
          self.velocities.pop()
      else:
        self.terminal |= item.x < 0
        remaining.append(item)
    self.items = remaining
    return reward

  @staticmethod
  def _touches(item, segment):
    x, y, end_x, end_y = segment
    dx, dy = end_x - x, end_y - y
    t = ((item.x - x) * dx + (item.y - y) * dy) / (dx * dx + dy * dy)
    t = min(1, max(0, t))
    return (item.x - x - t * dx) ** 2 + (item.y - y - t * dy) ** 2 < 4.5 ** 2

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    segments = list(self._segments())
    for index, (x, y, end_x, end_y) in enumerate(segments):
      points = np.linspace((x, y), (end_x, end_y), 32)
      visible = points[np.all((points >= 0) & (points < 100), axis=1)]
      cols, rows = (visible // 10).astype(int).T
      state[rows, cols, int(index == len(segments) - 1)] = True
    for item in self.items:
      if 0 <= item.x < 100:
        state[int(item.y // 10), int(item.x // 10), 2 + int(item.red)] = True
    state[int(self.next_y // 10), 9, 4] = True
    for row, value in enumerate((self.steps, len(self.angles), self.held,
                                 min(511, round(max(0, self.timer))), self.red_count)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
