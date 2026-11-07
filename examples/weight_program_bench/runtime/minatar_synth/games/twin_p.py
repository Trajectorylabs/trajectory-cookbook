from dataclasses import dataclass
from math import cos, hypot, pi, sin

import numpy as np


@dataclass
class Target:
  x: float
  y: float
  vx: float
  vy: float
  hazard: bool = False
  entered: bool = False


class TwinPEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.angle = 40., 50., 0.
    self.length = self.target_length = 15.
    self.targets = []
    self.timer = 0.
    self.until_hazard = 9
    self.multiplier = 1
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
    pin = previous == 5 and self.action != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(pin and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _endpoint(self):
    return self.x + cos(self.angle) * self.length, self.y + sin(self.angle) * self.length

  def _advance(self, pin):
    end_x, end_y = self._endpoint()
    if not (_inside(self.x, self.y) or _inside(end_x, end_y)):
      self.terminal = True
      return 0

    dx, dy = end_x - self.x, end_y - self.y
    start_x, start_y = self.x, self.y
    if pin:
      self.x, self.y = end_x, end_y
      self.angle += pi
      self.target_length = 15.
    elif self.action == 5:
      self.target_length = min(99, self.target_length + 0.2)
    self.angle = (self.angle + 0.05) % (2 * pi)
    self.length += (self.target_length - self.length) * 0.2

    self.timer -= 1
    if self.timer < 0:
      self._spawn()
    reward = 0
    remaining = []
    for target in self.targets:
      target.x += target.vx
      target.y += target.vy
      if -5 <= target.x < 105 and -5 <= target.y < 105:
        target.entered = True
      elif target.entered:
        if not target.hazard:
          self.multiplier = max(1, self.multiplier - 1)
        continue
      rx, ry = target.x - start_x, target.y - start_y
      along = min(1, max(0, (rx * dx + ry * dy) / (dx * dx + dy * dy)))
      if hypot(rx - along * dx, ry - along * dy) <= (3.5 if target.hazard else 3):
        if target.hazard:
          self.terminal = True
        else:
          reward += self.multiplier
          self.multiplier += 1
          continue
      remaining.append(target)
    self.targets = remaining
    return reward

  def _spawn(self):
    angle = self.random.uniform(0, 2 * pi)
    x, y = 50 + cos(angle) * 80, 50 + sin(angle) * 80
    dx, dy = self.random.uniform(20, 80, 2) - (x, y)
    scale = 0.2 / hypot(dx, dy)
    self.targets.append(Target(x, y, dx * scale, dy * scale, self.until_hazard == 0))
    self.until_hazard -= 1
    if self.until_hazard < 0:
      self.until_hazard = int(self.random.integers(8, 12))
    self.timer = self.random.uniform(60, 90)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    end_x, end_y = self._endpoint()
    _mark(state, self.x, self.y, 0)
    _mark(state, end_x, end_y, 1)
    for x, y in zip(np.linspace(self.x, end_x, 41), np.linspace(self.y, end_y, 41), strict=True):
      _mark(state, x, y, 2)
    for target in self.targets:
      _mark(state, target.x, target.y, 4 if target.hazard else 3)
    values = (self.steps, self.action, round(self.length * 4), round(self.target_length * 4),
              round(self.angle * 64), self.multiplier, self.until_hazard)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _inside(x, y):
  return 0 <= x < 99 and 0 <= y < 99


def _mark(state, x, y, channel):
  if 0 <= x < 100 and 0 <= y < 100:
    state[int(y // 10), int(x // 10), channel] = True
