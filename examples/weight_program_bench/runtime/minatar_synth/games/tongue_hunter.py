from dataclasses import dataclass
from math import atan2, ceil, cos, pi, sin

import numpy as np


@dataclass
class Prey:
  x: float
  y: float
  vx: float


@dataclass
class Hazard:
  x: float
  y: float = -5.


class TongueHunterEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.prey, self.hazards = [], []
    self.length, self.angle = 0., -pi / 2
    self.tip = (50., 90.)
    self.combo = 1
    self.steps = self.frames = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.frames % 60 == 0:
      left = self.random.random() < 0.5
      speed = self.random.uniform(0.3, 0.6)
      self.prey.append(Prey(-5 if left else 105, self.random.uniform(20, 60),
                            speed if left else -speed))
    if self.frames % 90 == 0:
      self.hazards.append(Hazard(self.random.uniform(10, 90)))

    dx, dy = cos(self.angle) * self.length, sin(self.angle) * self.length
    tip_x, tip_y = self.tip = (50 + dx, 90 + dy)
    if self.action == 5 and self.prey:
      target = min(self.prey, key=lambda p: (p.x - tip_x) ** 2 + (p.y - tip_y) ** 2)
      angle = atan2(target.y - 90, target.x - 50)
      delta = angle - self.angle
      if abs(delta) < 0.1:
        self.angle = angle
        self.length += 8
      else:
        self.angle += 0.1 if delta > 0 else -0.1
    self.length = min(80, max(0, self.length - 5))

    for prey in self.prey:
      prey.x += prey.vx * 1.05
    self.prey = [p for p in self.prey if -10 < p.x < 110]
    for hazard in self.hazards:
      hazard.y += 0.5
    self.hazards = [h for h in self.hazards if h.y < 110]

    reward = 0
    if self.length <= 5:
      self.combo = 1
    else:
      radius = 2.5 + 3 * (1 + sin(self.frames * 0.3) * 0.2)
      if any(abs(p.x - tip_x) <= radius and abs(p.y - tip_y) <= radius for p in self.prey):
        target = min(self.prey, key=lambda p: (p.x - tip_x) ** 2 + (p.y - tip_y) ** 2)
        self.prey.remove(target)
        reward = ceil(self.length) * self.combo
        self.combo += 1
      self.terminal = any(self._touches_tongue(h, dx, dy) for h in self.hazards)
    self.frames += 1
    return reward

  def _touches_tongue(self, hazard, dx, dy):
    radius = 1 + 4 * (1 + sin(self.frames * 0.15 + hazard.x) * 0.1)
    near, far = 0., 1.
    for offset, movement in ((hazard.x - 50, dx), (hazard.y - 90, dy)):
      if abs(movement) < 1e-9:
        if abs(offset) > radius:
          return False
      else:
        edges = sorted(((offset - radius) / movement, (offset + radius) / movement))
        near, far = max(near, edges[0]), min(far, edges[1])
    return near <= far

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 5, 0] = True
    if self.length > 5:
      tip_x, tip_y = self.tip
      for x, y in zip(np.linspace(50, tip_x, 33), np.linspace(90, tip_y, 33), strict=True):
        self._mark(state, x, y, 1)
      self._mark(state, tip_x, tip_y, 2)
    for prey in self.prey:
      self._mark(state, prey.x, prey.y, 3)
    for hazard in self.hazards:
      self._mark(state, hazard.x, hazard.y, 4)
    values = (self.steps, self.action, self.combo, round(self.length * 4),
              round((self.angle + pi) * 128))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state

  @staticmethod
  def _mark(state, x, y, channel):
    if 0 <= x < 100 and 0 <= y < 100:
      state[int(y // 10), int(x // 10), channel] = True
