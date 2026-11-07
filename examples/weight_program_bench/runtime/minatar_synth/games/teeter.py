from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass
class Bar:
  x: float
  y: float
  width: float
  angle: float


@dataclass
class Floor:
  x: float
  width: float
  score: int
  change: int = 0


class TeeterEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.bars = []
    for _ in range(7):
      self._add_bar()
    self.floors = []
    self.ball = self.target = None
    self.sign, self.steps = 1, 0
    self.held = self.terminal = False

  def _add_bar(self):
    for _ in range(99):
      width = self.random.uniform(12, 18)
      angle = self.random.uniform(0.2, 0.8) * self.random.choice((-1, 1))
      margin = 5 + width * cos(angle) / 2
      x, y = self.random.uniform(margin, 100 - margin), self.random.uniform(20, 70)
      if all(abs(x - bar.x) >= (width + bar.width) / 2 + 5 or
             abs(y - bar.y) >= 6 + (width * abs(sin(angle)) +
                                    bar.width * abs(sin(bar.angle))) / 2 for bar in self.bars):
        self.bars.append(Bar(x, y, width, angle))
        return True
    return False

  def _drop(self):
    for _ in range(99):
      x = self.random.uniform(20, 80)
      if any(abs(bar.x - x) < bar.width * cos(bar.angle) / 2 for bar in self.bars):
        break
    self.target = np.array([x, -4.])
    self.ball = self.target.copy()
    self.floors = []
    edge = 0.
    while edge < 100:
      width = self.random.uniform(20, 30)
      score = int(np.prod(self.random.uniform(1, 3.1, 2)))
      self.floors.append(Floor(edge + width / 2, width, score))
      edge += width
    for sign in (1, -1):
      floor = self.floors[self.random.integers(len(self.floors))]
      floor.score = 0
      floor.change = sign * int(np.prod(self.random.uniform(1, 2.2, 2)))
    for floor in self.floors:
      floor.x -= (edge - 100) / 2

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.sign *= -1
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.ball is None:
      self._drop()
    self.target[1] += 0.5
    self.ball += (self.target - self.ball) * 0.5
    x, y = self.ball
    for bar in self.bars:
      angle = bar.angle * self.sign
      dx, dy = cos(angle), sin(angle)
      ox, oy = x - bar.x, y - bar.y
      if abs(ox * dx + oy * dy) < bar.width / 2 + 3 and abs(oy * dx - ox * dy) < 4.5:
        reach = (bar.width / 2 + 7) * (1 if angle > 0 else -1)
        self.target[:] = bar.x + reach * dx, bar.y + reach * dy
    reward = change = 0
    remaining = []
    for floor in self.floors:
      if abs(floor.x - x) < floor.width / 2 + 2.5 and abs(y - 95) < 8:
        reward += floor.score
        change += floor.change
      else:
        remaining.append(floor)
    self.floors = remaining
    if change > 0:
      for _ in range(change):
        reward += int(not self._add_bar())
    elif change < 0:
      for _ in range(min(-change, len(self.bars))):
        self.bars.pop(self.random.integers(len(self.bars)))
    if y > 99:
      self.terminal = not self.bars
      self.ball = self.target = None
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    if self.ball is not None:
      x, y = np.clip(self.ball // 10, 0, 9).astype(int)
      state[y, x, 0] = True
    for bar in self.bars:
      angle = bar.angle * self.sign
      offset = np.linspace(-bar.width / 2, bar.width / 2, 20)
      cols = ((bar.x + offset * cos(angle)) // 10).astype(int)
      rows = ((bar.y + offset * sin(angle)) // 10).astype(int)
      state[rows, cols, 1] = True
    columns = np.arange(5, 100, 10)
    for floor in self.floors:
      channel = 2 if floor.score else 3 if floor.change > 0 else 4
      state[9, :, channel] |= abs(columns - floor.x) < floor.width / 2
    for index, floor in enumerate(sorted(self.floors, key=lambda floor: floor.x)):
      value = floor.score or abs(floor.change)
      for bit in range(4):
        state[index, bit, 5] = bool(value & (1 << bit))
    for row, value in enumerate((self.steps, self.sign > 0, len(self.bars), self.held)):
      for bit in range(9):
        state[6 + row, bit, 5] = bool(value & (1 << bit))
    return state
