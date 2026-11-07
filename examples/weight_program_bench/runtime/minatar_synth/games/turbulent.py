from bisect import bisect_right
from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, remainder, sin

import numpy as np


@dataclass
class Wave:
  x: float
  height: float
  phase: float
  speed: float


@dataclass
class Mine:
  x: float
  vx: float = 0.


class TurbulentEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.waves = [
      Wave(i * 20 - 20, self.random.uniform(10, 30),
           (i % 2) * pi + self.random.uniform(-pi / 4, pi / 4),
           self.random.uniform(0.01, 0.02))
      for i in range(7)
    ]
    self.mines = []
    self.mine_distance = 0.
    self.x, self.y = 40., 60.
    self.vx = self.vy = self.angle = self.jump_x = 0.
    self.airborne = self.terminal = False
    self.steps = self.action = 0
    self._surface_points()

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

  def _surface_points(self):
    peaks = sorted((w.x, 60 + sin(w.phase) * w.height) for w in self.waves)
    points = [peaks[0]]
    for (x, y), (right, next_y) in zip(peaks, peaks[1:]):
      points.extend((x + 5 * i, y + (next_y - y) * ratio)
                    for i, ratio in enumerate((0.2, 0.5, 0.8), 1))
      points.append((right, next_y))
    self.surface_x, self.surface_y = zip(*points, strict=True)

  def _surface(self, x):
    index = min(len(self.surface_x) - 2, max(0, bisect_right(self.surface_x, x) - 1))
    width = self.surface_x[index + 1] - self.surface_x[index]
    rise = self.surface_y[index + 1] - self.surface_y[index]
    height = self.surface_y[index] + rise * (x - self.surface_x[index]) / width
    return height, rise, width

  def _advance(self, jump):
    scroll = 0.1 + max(0, self.x - 50) * 0.1
    for wave in self.waves:
      wave.x -= scroll
      if wave.x < -20:
        wave.x += 140
        wave.height = self.random.uniform(10, 30)
        wave.phase = self.random.uniform(0, 2 * pi)
        wave.speed = self.random.uniform(0.01, 0.02)
      wave.phase += wave.speed
    self._surface_points()

    self.mine_distance -= scroll
    if self.mine_distance < 0:
      self.mines.append(Mine(103))
      self.mine_distance = self.random.uniform(100, 120)
    mine_positions = []
    remaining = []
    for mine in self.mines:
      mine.x -= scroll
      if not self.surface_x[0] <= mine.x < self.surface_x[-1]:
        continue
      height, rise, width = self._surface(mine.x)
      mine.vx = (mine.vx + rise * 0.001) * 0.9
      mine.x += mine.vx
      if mine.x >= -3:
        remaining.append(mine)
        mine_positions.append((mine.x, height + rise * mine.vx / width - 5))
    self.mines = remaining

    if self.airborne:
      self.jump_x -= scroll
      self.vx = (self.vx + 0.005) * 0.98
      self.vy = (self.vy + (0.02 if self.action == 5 else 0.1)) * 0.98
      self.x += self.vx
      self.y += self.vy
      angle = atan2(self.vy, self.vx)
    else:
      previous_x, previous_y = self.x, self.y
      height, rise, width = self._surface(self.x)
      self.vx = (self.vx + rise * 0.002) * 0.925 + 0.025
      self.x += self.vx
      self.y = height + rise * self.vx / width
      angle = atan2(rise, width)
      if jump:
        self.jump_x = self.x
        self.vx = (self.x - previous_x) * 2 + sin(angle)
        self.vy = min(-1, (self.y - previous_y) * 5 - cos(angle))
        self.x += self.vx
        self.y += self.vy
        self.airborne = True

    self.x = min(95, max(5, self.x - scroll))
    self.y = min(95, max(5, self.y))
    self.angle += remainder(angle - self.angle, 2 * pi) * 0.1
    nose_x = self.x + 5.8 * sin(self.angle)
    nose_y = self.y - 5.8 * cos(self.angle)
    reward = 0
    if self.airborne and nose_y >= self._surface(nose_x)[0] - 2:
      reward = max(0, ceil(self.x - self.jump_x))
      self.airborne = False
      self.vx *= 0.5
    self.terminal = any(hypot(x - nose_x, y - nose_y) <= 4 for x, y in mine_positions)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), int(self.x // 10), 0] = True
    for col in range(10):
      heights = [self._surface(col * 10 + offset)[0] for offset in (0, 5, 10)]
      state[int(min(heights) // 10):int(max(heights) // 10) + 1, col, 1] = True
    for mine in self.mines:
      row = int((self._surface(mine.x)[0] - 5) // 10)
      state[row, int(np.clip(mine.x // 10, 0, 9)), 2] = True
    if self.airborne:
      state[int(self.y // 10), int(self.x // 10), 3] = True
      state[9, int(np.clip(self.jump_x // 10, 0, 9)), 4] = True
    values = (self.steps, self.action, round((self.vx + 8) * 16),
              round((self.vy + 16) * 8), round((self.angle + pi) * 64))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
