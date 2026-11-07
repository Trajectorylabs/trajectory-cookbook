from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Wall:
  y: float
  gaps: list[tuple[float, float]]

  def segments(self):
    left = 0.0
    for start, end in self.gaps:
      yield left, start
      left = end
    yield left, 100.0


class PlasmaSplitterEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.anchor = 50.0, 0.0
    self.direction = 1
    self.speed, self.distance = 0.2, 0.0
    self.steps = 0
    self.pressed = self.terminal = False
    self.walls: list[Wall] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if self.pressed and not pressed:
      self.anchor = self.x
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self):
    count = int(self.random.choice((1, 2, 2, 3, 3, 3)))
    gaps = []
    x = 0.0
    for index in range(count):
      x += self.random.uniform(10 if index else 0, 76 - x - (count - index - 1) * 34)
      gaps.append((x, x + 24))
      x += 24
    self.walls.append(Wall(-3, gaps))
    self.distance = self.random.uniform(50, 80)

  def _advance(self):
    if self.pressed:
      self.speed += 0.01
    else:
      self.speed += (0.2 - self.speed) * 0.1
      self.x += self.direction
      if self.x < 4 or self.x > 96:
        self.direction *= -1
    self.distance -= self.speed
    if self.distance < 0:
      self._spawn()
    reward = 0
    remaining = []
    low, high = sorted((self.x, self.anchor))
    for wall in self.walls:
      wall.y += self.speed
      destroyed = False
      for index, (left, right) in enumerate(wall.segments()):
        if right <= left:
          continue
        endpoint_hit = any(left < x + 4 and x - 4 < right for x in (self.x, self.anchor))
        if 89 < wall.y < 100 and endpoint_hit:
          self.terminal = True
        elif index < len(wall.gaps) and 91 < wall.y < 96 and low < right and left < high:
          reward += math.floor(self.speed / 0.2 * (right - left))
          destroyed = True
      if not destroyed and wall.y <= 103:
        remaining.append(wall)
    self.walls = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    a, b = int(self.x // 10), int(self.anchor // 10)
    state[9, a, 0] = True
    state[9, b, 1] = True
    state[9, min(a, b):max(a, b) + 1, 2] = True
    columns = np.arange(5, 100, 10)
    for wall in self.walls:
      if 0 <= wall.y < 100:
        row = int(wall.y // 10)
        for start, end in wall.gaps:
          state[row, :, 4] |= (columns >= start) & (columns < end)
        for start, end in wall.segments():
          state[row, :, 3] |= (columns >= start) & (columns < end)
    for row, value in enumerate((self.steps, round(self.x * 4), round(self.anchor * 4),
                                 min(511, round(self.speed * 30)))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
