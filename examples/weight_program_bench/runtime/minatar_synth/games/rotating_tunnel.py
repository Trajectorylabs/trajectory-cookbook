from dataclasses import dataclass
from math import ceil, cos, pi, sin

import numpy as np


@dataclass
class Arc:
  radius: float
  start: float
  width: float
  gold: bool


class RotatingTunnelEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = 0.
    self.radius = 20.
    self.multiplier = 0.
    self.steps = self.spawn_timer = self.score_timer = 0
    self.held = self.terminal = False
    self.arcs = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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

  def _advance(self):
    self.angle = (self.angle + 0.025) % (2 * pi)
    self.radius = min(50, max(10, self.radius + (0.65 if self.held else -0.65)))
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      gold = self.random.random() < 0.3
      self.arcs.append(Arc(50, self.random.uniform(0, 2 * pi),
                           self.random.uniform(pi / 4, pi / 3), gold))
      self.spawn_timer += 80
    reward = 0
    for arc in self.arcs:
      arc.radius -= 0.25
      if arc.radius <= 10 or abs(self.radius - arc.radius) > 2.5:
        continue
      angle = (self.angle - arc.start) % (2 * pi)
      margin = 2.5 / arc.radius
      if angle > arc.width + margin and angle < 2 * pi - margin:
        continue
      if arc.gold:
        self.score_timer -= 1
        if self.score_timer < 0:
          self.multiplier += 1
          reward += ceil(self.multiplier)
          self.score_timer += 5
      else:
        self.terminal = True
    self.arcs = [arc for arc in self.arcs if arc.radius > 10]
    self.multiplier *= 0.99
    return reward

  @staticmethod
  def _mark(state, x, y, channel):
    cols = np.clip(np.asarray(x) // 10, 0, 9).astype(int)
    rows = np.clip(np.asarray(y) // 10, 0, 9).astype(int)
    state[rows, cols, channel] = True

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    dx, dy = cos(self.angle), sin(self.angle)
    self._mark(state, 50 + self.radius * dx, 50 + self.radius * dy, 0)
    for arc in self.arcs:
      angles = np.linspace(arc.start, arc.start + arc.width, 40)
      self._mark(state, 50 + arc.radius * np.cos(angles), 50 + arc.radius * np.sin(angles),
                 2 if arc.gold else 1)
    angles = np.linspace(0, 2 * pi, 96)
    for radius in (10, 50):
      self._mark(state, 50 + radius * np.cos(angles), 50 + radius * np.sin(angles), 3)
    future = self.angle + 0.1
    self._mark(state, 50 + self.radius * cos(future), 50 + self.radius * sin(future), 4)
    values = (self.steps, self.held, self.spawn_timer, self.score_timer,
              round(self.radius * 10), round(self.angle * 100),
              min(1023, round(self.multiplier * 100)))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
