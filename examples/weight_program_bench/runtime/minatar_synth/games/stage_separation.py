from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float
  vy: float
  size: float


class StageSeparationEnvironment:
  score_cap = 1500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.vy = 90., -0.5
    self.stage, self.burn = 4., 100
    self.camera, self.zoom = 0., 1.
    self.distance = 99.
    self.steps = 0
    self.held = self.terminal = False
    self.stages, self.debris = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0.
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, tap):
    reward = -self.vy * 0.1
    self.camera += max(-99, min(0, (self.y - 130 * self.zoom - self.camera) * 0.1))
    self.zoom = min(3, max(1, 1 - self.y / 9999))
    self.vy += 0.05 - self.burn / 600
    self.y += self.vy
    self.burn = max(0, self.burn - 1)
    if tap and self.stage > 2:
      self.stages.append(Body(50, self.y + 10 * (self.stage - 1),
                              self.random.uniform(-0.5, 0.5), self.vy + 0.5, 10))
      self.stage -= 1
      self.vy -= 0.5
      self.burn = 60
    self.stage = min(4, self.stage + 0.02)
    self.terminal = self.y - self.camera > 150 * self.zoom
    for stage in self.stages:
      stage.vy += 0.1
      stage.x += stage.vx
      stage.y += stage.vy
    self.stages = [stage for stage in self.stages if stage.y <= self.camera + 200 * self.zoom]
    self.distance += self.vy
    if self.distance < 0:
      side = self.random.choice((-1, 1))
      self.debris.append(Body(50 + side * 70 * self.zoom,
                              self.camera - self.random.uniform(0, 200) * self.zoom,
                              -side * self.random.uniform(0.5, 0.8) * self.zoom / 2,
                              self.random.uniform(-15, 0) * self.zoom / 2,
                              self.random.uniform(5, 15)))
      self.distance = self.random.uniform(120, 160) / sqrt(self.zoom)
    for debris in self.debris:
      debris.x += debris.vx
      debris.y += debris.vy
      self.terminal |= self._hits(debris)
    self.debris = [debris for debris in self.debris if abs(debris.x - 50) <= 70 * self.zoom]
    return reward

  def _hits(self, debris):
    dx = abs(debris.x - 50)
    dy = max(self.y - debris.y, debris.y - (self.y + self.stage * 10), 0)
    near = sqrt(max(0, dx - 5) ** 2 + dy ** 2)
    far = sqrt((dx + 5) ** 2 + max(abs(debris.y - self.y),
                                  abs(debris.y - self.y - self.stage * 10)) ** 2)
    return near <= debris.size + 2 and far >= debris.size - 2

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    rows, cols = np.ogrid[7.5:150:15, 5:100:10]
    world_x = 50 + (cols - 50) * self.zoom
    world_y = self.camera + rows * self.zoom
    for x, y, height, channel in [(50, self.y, self.stage * 10, 0),
                                   *((s.x, s.y, s.size, 1) for s in self.stages)]:
      state[:, :, channel] |= (abs(world_x - x) < 5 + 5 * self.zoom) & (
        (world_y > y - 7.5 * self.zoom) & (world_y < y + height + 7.5 * self.zoom))
    for debris in self.debris:
      state[:, :, 2] |= abs(np.hypot(world_x - debris.x, world_y - debris.y) - debris.size) < 8 * self.zoom
    if self.burn:
      y = (self.y + 10 * self.stage - self.camera) / self.zoom
      if 0 <= y < 150:
        state[int(y // 15), 5, 3] = True
    state[9, :, 4] = True
    values = (self.steps, round(self.stage * 100), self.burn,
              min(2047, max(0, round((self.vy + 100) * 10))), self.held)
    for row, value in enumerate(values):
      for bit in range(11):
        state[row + bit // 10 * 5, bit % 10, 5] = bool(value & (1 << bit))
    return state
