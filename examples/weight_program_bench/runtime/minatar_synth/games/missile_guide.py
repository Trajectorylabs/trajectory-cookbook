from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Outpost:
  x: float
  y: float = -5
  phase: float = 0


@dataclass
class Wall:
  x: float
  y: float = -10


class MissileGuideEnvironment:
  score_cap = 120

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 65.0
    self.last_x = 50.0
    self.fuel = 1000
    self.frames = self.steps = self.combo = 0
    self.timer = 30
    self.frozen = self.terminal = False
    self.outposts: list[Outpost] = []
    self.walls: list[Wall] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.frozen = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _target(self):
    return min((o for o in self.outposts if -60 < o.y - self.y < 40),
               key=lambda o: abs(o.y - self.y), default=None)

  def _advance(self):
    self.fuel = self.fuel - 4 if self.frozen else min(1000, self.fuel + 1)
    if self.fuel <= 0:
      self.terminal = True
      return 0
    self.timer -= 1
    if not self.timer:
      x = min(88, max(12, self.last_x + self.random.uniform(-40, 40)))
      if self.random.random() < 0.8:
        self.outposts.append(Outpost(x, phase=self.random.uniform(0, math.tau)))
        self.last_x = x
      if self.frames > 60 and self.random.random() < 0.6:
        self.walls.append(Wall(min(80, max(20, x + self.random.uniform(-20, 20)))))
      self.timer = 50
    self.frames += 1
    if not self.frozen:
      target = self._target()
      if target is not None:
        self.x += min(0.8, max(-0.8, (target.x - self.x) * 0.03))
        self.y += min(0.8, max(-0.8, (target.y - self.y) * 0.01))
      self.x = min(94, max(6, self.x))
      self.y = min(90, max(60, self.y))
    reward = 0
    remaining = []
    for outpost in self.outposts:
      outpost.y += 0.6
      outpost.phase += 0.15
      radius = 3 * (1 + math.sin(outpost.phase) * 0.15)
      if not self.frozen and abs(outpost.x - self.x) < radius + 2 and abs(outpost.y - self.y) < radius + 3.5:
        self.combo += 1
        reward += 1 + self.combo // 3
        self.fuel = min(1000, self.fuel + 120)
      elif outpost.y > 105:
        self.combo = 0
      else:
        remaining.append(outpost)
    self.outposts = remaining
    for wall in self.walls:
      wall.y += 0.6
      if not self.frozen and abs(wall.y - self.y) < 6 and abs(wall.x - self.x) > 13:
        self.terminal = True
    self.walls = [wall for wall in self.walls if wall.y <= 105]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y // 10)), int(self.x // 10), 0] = True
    for outpost in self.outposts:
      if 0 <= outpost.y < 100:
        state[int(outpost.y // 10), int(outpost.x // 10), 1] = True
    for wall in self.walls:
      if 0 <= wall.y < 100:
        row = int(wall.y // 10)
        state[row, :, 2] |= abs(np.arange(5, 100, 10) - wall.x) >= 15
        state[row, int(wall.x // 10), 3] = True
    target = self._target()
    if target is not None and 0 <= target.y < 100:
      state[int(target.y // 10), int(target.x // 10), 4] = True
    for row, value in enumerate((self.steps, self.fuel // 2, self.combo, round(self.x * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.frozen
    return state
