from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
STARTS = ((50, 29), (71, 50), (50, 71), (29, 50))


@dataclass
class Bar:
  x: float
  y: float
  length: float
  angle: float
  side: int
  pivot: float = 0

  def advance(self):
    self.angle = (self.angle + 0.03) % 4
    if not self.side <= self.angle < self.side + pi / 4:
      return
    self.angle = self.side
    dx, dy = DIRECTIONS[self.side]
    distance = self.length * (1 - self.pivot)
    room = (71 - self.x, 71 - self.y, self.x - 29, self.y - 29)[self.side]
    if distance > room:
      self.x += room * dx
      self.y += room * dy
      self.pivot = 1 - (distance - room) / self.length
      self.side = (self.side + 1) % 4
    else:
      self.x += distance * dx
      self.y += distance * dy
      self.angle = (self.angle + 2) % 4
      self.pivot = 0

  def points(self):
    distance = np.linspace(-self.pivot, 1 - self.pivot, 32) * self.length
    angle = self.angle * pi / 2
    return self.x + distance * cos(angle), self.y + distance * sin(angle)


@dataclass
class Item:
  x: float
  y: float
  spike: bool
  timer: int
  size: int = 0


class SquareBarEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.player = Bar(50, 29, 9, 3, 0)
    self.enemy = Bar(50, 71, 12, 1, 2)
    self.items = [self._item(i < 3) for i in range(10)]
    self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False

  def _item(self, spike):
    for _ in range(99):
      x = self.random.uniform(9, 70) - 50
      y = self.random.uniform(9, 15 if spike else 26) - 50
      dx, dy = DIRECTIONS[self.random.integers(4)]
      px, py = 50 + dx * x - dy * y, 50 + dy * x + dx * y
      if hypot(px - self.player.x, py - self.player.y) > 36:
        break
    return Item(px, py, spike, int(self.random.integers(777, 999)))

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

  def _chased(self):
    difference = atan2(self.player.y - 50, self.player.x - 50) - atan2(
      self.enemy.y - 50, self.enemy.x - 50)
    return (difference + pi) % (2 * pi) - pi > 0

  def _advance(self):
    chased = self._chased()
    target, rate = (29, 0.1) if self.held else (9, 0.2)
    self.player.length += (target - self.player.length) * rate
    self.enemy.length = 15
    self.player.advance()
    self.enemy.advance()
    px, py = self.player.points()
    ex, ey = self.enemy.points()
    reward = 0
    if np.any(np.hypot(px[:, None] - ex, py[:, None] - ey) < 3):
      if chased:
        self.terminal = True
      else:
        reward += 10 * self.multiplier
        self.multiplier += 1
        side = (self.player.side + 2) % 4
        self.enemy = Bar(*STARTS[side], 12, (side - 1) % 4, side)
    for index, item in enumerate(self.items):
      if item.timer % 15 == 0:
        item.size = max(0, item.size - 1) if item.timer < 60 else min(4, item.size + 1)
      reach = item.size / 2 + 1.5
      hit = item.size > 0 and np.any((abs(px - item.x) < reach) & (abs(py - item.y) < reach))
      if hit and not item.spike:
        reward += self.multiplier
        self.items[index] = self._item(False)
      else:
        self.terminal |= bool(hit and item.size == 4)
        item.timer -= 1
        if item.timer < 0:
          self.items[index] = self._item(item.spike)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, bar in ((0, self.player), (1, self.enemy)):
      x, y = bar.points()
      cols = np.clip(x // 10, 0, 9).astype(int)
      rows = np.clip(y // 10, 0, 9).astype(int)
      state[rows, cols, channel] = True
    for item in self.items:
      if item.size:
        channel = 2 if not item.spike else 3 if item.size == 4 else 4
        state[int(item.y // 10), int(item.x // 10), channel] = True
    values = (self.steps, self.multiplier, self.held, self._chased(),
              round(self.player.length * 10), self.player.side, self.enemy.side)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
