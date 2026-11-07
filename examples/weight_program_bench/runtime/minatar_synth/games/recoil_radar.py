from dataclasses import dataclass
from math import atan2, cos, floor, sin

import numpy as np


@dataclass
class Hazard:
  position: np.ndarray
  velocity: np.ndarray
  size: float


class RecoilRadarEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.position = np.array([50.0, 50.0])
    self.velocity = np.zeros(2)
    self.charge = self.steps = self.shot_ticks = 0
    self.radius = 8.0
    self.timer = 45
    self.held = self.terminal = False
    self.hazards = []
    self.shot = None

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance(held)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, held):
    reward = 0
    if held:
      self.charge += 1
      self.radius = min(46, 8 + self.charge * 0.55)
    elif self.held:
      target_index = None
      distance = self.radius
      if self.charge >= 10:
        for index, hazard in enumerate(self.hazards):
          candidate = np.linalg.norm(self.position - hazard.position)
          if candidate <= distance:
            target_index, distance = index, candidate
      if target_index is not None:
        target = self.hazards.pop(target_index)
        delta = self.position - target.position
        angle = atan2(delta[1], delta[0])
        self.velocity += (1.2 + (46 - distance) * 0.025) * np.array([cos(angle), sin(angle)])
        self.shot = (self.position.copy(), target.position.copy())
        self.shot_ticks = 5
        reward = floor(55 - distance + 0.5)
      self.charge, self.radius = 0, 8.0
    self.held = held
    self.position += self.velocity
    self.velocity *= 0.985
    lower = np.array([8, 9])
    self.velocity = np.where(self.position < lower, abs(self.velocity) * 0.8, self.velocity)
    self.velocity = np.where(self.position > 92, -abs(self.velocity) * 0.8, self.velocity)
    np.clip(self.position, lower, 92, out=self.position)
    self.timer -= 1
    if self.timer < 0:
      self._spawn()
      self.timer += 72
    angle = atan2(self.velocity[1], self.velocity[0])
    direction = np.array([cos(angle), sin(angle)])
    bar = self.position + np.linspace(-4.5, 4.5, 10)[:, None] * direction
    remaining = []
    for hazard in self.hazards:
      hazard.position += hazard.velocity * 0.63
      half = hazard.size / 2
      body_hit = np.all(abs(self.position - hazard.position) < half + 3)
      bar_hit = np.any(np.all(abs(bar - hazard.position) < half + 1, axis=1))
      self.terminal |= bool(body_hit or bar_hit)
      if np.all((-12 <= hazard.position) & (hazard.position <= 112)):
        remaining.append(hazard)
    self.hazards = remaining
    self.shot_ticks = max(0, self.shot_ticks - 1)
    return reward

  def _spawn(self):
    offset = self.random.uniform(10, 90)
    edge = self.random.integers(4)
    position = np.array(((offset, -5), (105, offset), (offset, 105), (-5, offset))[edge])
    delta = self.position - position
    angle = atan2(delta[1], delta[0]) + self.random.uniform(-0.45, 0.45)
    self.hazards.append(Hazard(position, np.array([cos(angle), sin(angle)]), self.random.uniform(5, 8)))

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = (self.position // 10).astype(int)
    state[y, x, 0] = True
    for hazard in self.hazards:
      hx, hy = (hazard.position // 10).astype(int)
      if 0 <= hx < 10 and 0 <= hy < 10:
        revealed = self.charge >= 10 and np.linalg.norm(hazard.position - self.position) <= self.radius
        state[hy, hx, 1 + int(revealed)] = True
    if self.held:
      rows, cols = np.ogrid[5:100:10, 5:100:10]
      state[:, :, 3] = np.hypot(cols - self.position[0], rows - self.position[1]) <= self.radius
    if self.shot_ticks:
      points = np.linspace(*self.shot, 50)
      xs, ys = np.clip(points // 10, 0, 9).astype(int).T
      state[ys, xs, 4] = True
    gauges = (self.steps, min(511, self.charge), self.timer, self.held,
              int(round(self.velocity[0] * 16)) + 256, int(round(self.velocity[1] * 16)) + 256)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
