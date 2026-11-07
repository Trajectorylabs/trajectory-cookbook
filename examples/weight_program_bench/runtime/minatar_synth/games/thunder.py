from dataclasses import dataclass
from math import ceil, cos, hypot, pi, sin

import numpy as np


@dataclass
class Bolt:
  x: float
  y: float
  angle: float
  speed: float
  ticks: int
  parent: "Bolt | None" = None
  active: bool = False

  def __post_init__(self):
    self.origin = (self.x, self.y)


@dataclass
class Star:
  x: float
  y: float
  vy: float


class ThunderEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.bolts = []
    self.stars = []
    self.active_ticks = -1
    self.x, self.vx = 40., 1
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
    turn = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(turn and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, turn):
    if not self.bolts:
      self.bolts.append(Bolt(self.random.uniform(30, 70), 0, pi / 2, 0.5, 30))
    self.active_ticks -= 1
    remaining = []
    for bolt in self.bolts:
      if bolt.active:
        if self.active_ticks >= 0:
          remaining.append(bolt)
        continue
      bolt.ticks -= 1
      if self.active_ticks > 0:
        if bolt.ticks > 0:
          self.stars.append(Star(bolt.x, bolt.y, -bolt.y * 0.02))
        continue
      remaining.append(bolt)
      if bolt.ticks > 0:
        bolt.x += cos(bolt.angle) * bolt.speed
        bolt.y += sin(bolt.angle) * bolt.speed
        if self.active_ticks < 0 and (bolt.y > 90 or len(self.bolts) > 160):
          self._strike(bolt)
      elif bolt.ticks == 0:
        for _ in range(self.random.integers(1, 4)):
          remaining.append(Bolt(
            bolt.x, bolt.y, bolt.angle + self.random.uniform(-0.7, 0.7),
            self.random.uniform(0.3, 1), ceil(self.random.uniform(20, 40)), bolt,
          ))
    self.bolts = remaining
    if turn or (self.x < 0 and self.vx < 0) or (self.x > 99 and self.vx > 0):
      self.vx *= -1
    self.x += self.vx
    self.terminal = any(
      bolt.active and _segment_distance(self.x, 87, bolt) <= 5
      for bolt in self.bolts
    )
    if self.terminal:
      return 0
    reward = 0
    stars = []
    for star in self.stars:
      star.vy += 0.1
      star.y += star.vy
      if star.y > 89 and star.vy > 0:
        star.vy *= -0.5
        if star.vy > -0.1:
          continue
      if abs(star.x - self.x) <= 5 and abs(star.y - 87) <= 5:
        reward += self.multiplier
        self.multiplier += 1
      else:
        stars.append(star)
    self.stars = stars
    return reward

  def _strike(self, bolt):
    self.active_ticks = 20
    self.multiplier = 1
    for _ in range(99):
      bolt.active = True
      bolt = bolt.parent
      if bolt is None:
        break

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, int(np.clip(self.x // 10, 0, 9)), 0] = True
    for bolt in self.bolts:
      start_x, start_y = bolt.origin
      samples = ceil(hypot(bolt.x - start_x, bolt.y - start_y) / 3) + 1
      for x, y in zip(np.linspace(start_x, bolt.x, samples),
                      np.linspace(start_y, bolt.y, samples), strict=True):
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), 2 if bolt.active else 1] = True
    for star in self.stars:
      if 0 <= star.x < 100 and 0 <= star.y < 100:
        state[int(star.y // 10), int(star.x // 10), 3] = True
    state[9, :, 4] = True
    values = (self.steps, self.action, self.vx > 0, self.multiplier, self.active_ticks + 1)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _segment_distance(x, y, bolt):
  start_x, start_y = bolt.origin
  dx, dy = bolt.x - start_x, bolt.y - start_y
  length_squared = dx * dx + dy * dy
  if length_squared == 0:
    return hypot(x - start_x, y - start_y)
  fraction = min(1, max(0, ((x - start_x) * dx + (y - start_y) * dy) / length_squared))
  return hypot(x - start_x - fraction * dx, y - start_y - fraction * dy)
