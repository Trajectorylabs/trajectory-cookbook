from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass
class Critter:
  x: float
  y: float
  angle: float


@dataclass
class Spike:
  x: float
  y: float
  warm: int


class RopeCorralEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 30., 50.
    self.bx, self.by = 20., 50.
    self.vx, self.vy = 0.75, 0.32
    self.angle = self.swept = 0.
    self.steps = self.frame = self.hitstop = 0
    self.timer = 200
    self.held = self.terminal = False
    self.critters = [self._critter() for _ in range(5)]
    for critter in self.critters:
      if hypot(critter.x - self.x, critter.y - self.y) < 18:
        critter.x = 15 + (critter.x + 25) % 70
    self.spikes = []
    for _ in range(3):
      self._spawn_spike(0)

  def _critter(self):
    x, y = self.random.uniform(15, 85, 2)
    return Critter(x, y, self.random.uniform(0, 2 * pi))

  def _spawn_spike(self, warm):
    for _ in range(8):
      x, y = self.random.uniform(10, 90, 2)
      if hypot(x - self.x, y - self.y) >= 20 and all(
        hypot(x - spike.x, y - spike.y) >= 24 for spike in self.spikes
      ):
        break
    self.spikes.append(Spike(x, y, warm))

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    frozen = self.hitstop > 0
    self.hitstop = max(0, self.hitstop - 1)
    reward = 0
    if self.held:
      if tap:
        self.angle = atan2(self.y - self.by, self.x - self.bx)
        self.swept = 0
      if not frozen:
        self.angle += 0.075
        self.swept += 0.075
      dx, dy = cos(self.angle), sin(self.angle)
      self.x, self.y = self.bx + 22 * dx, self.by + 22 * dy
      self.vx, self.vy = -1.32 * dy, 1.32 * dx
      if self.swept >= 2 * pi:
        self.swept = 0
        count = len(self.critters) + len(self.spikes)
        self.critters = [c for c in self.critters if hypot(c.x - self.bx, c.y - self.by) >= 20]
        self.spikes = [s for s in self.spikes if hypot(s.x - self.bx, s.y - self.by) >= 20]
        caught = count - len(self.critters) - len(self.spikes)
        if caught:
          reward += 4 * caught ** 2
          self.hitstop = min(8, 2 + caught)
          self.critters.extend(self._critter() for _ in range(caught))
    elif not frozen:
      self.x += self.vx
      self.y += self.vy
      if not 6 <= self.x <= 94:
        self.vx *= -1
        self.x = min(94, max(6, self.x))
      if not 6 <= self.y <= 94:
        self.vy *= -1
        self.y = min(94, max(6, self.y))
      dx, dy = self.x - self.bx, self.y - self.by
      distance = hypot(dx, dy)
      if distance > 22:
        self.bx, self.by = self.x - 22 * dx / distance, self.y - 22 * dy / distance
    if not frozen:
      self.timer -= 1
      if self.timer < 0 and len(self.critters) < 8:
        self.critters.append(self._critter())
        self.timer = self.random.uniform(140, 220)
    if self.frame % 700 == 699 and len(self.spikes) < 6:
      self._spawn_spike(60)
    remaining = []
    for critter in self.critters:
      critter.angle += self.random.uniform(-0.2, 0.2)
      critter.x = min(92, max(8, critter.x + 0.28 * cos(critter.angle)))
      critter.y = min(92, max(8, critter.y + 0.28 * sin(critter.angle)))
      if critter.x in (8, 92):
        critter.angle = pi - critter.angle
      if critter.y in (8, 92):
        critter.angle *= -1
      if hypot(critter.x - self.x, critter.y - self.y) >= 4.5:
        remaining.append(critter)
    collected = len(self.critters) - len(remaining)
    self.critters = remaining + [self._critter() for _ in range(collected)]
    for spike in self.spikes:
      if spike.warm:
        spike.warm -= 1
      elif hypot(spike.x - self.x, spike.y - self.y) < 4:
        self.terminal = True
    self.frame += 1
    return reward + collected

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    positions = [(self.x, self.y, 0), (self.bx, self.by, 1)]
    positions.extend((c.x, c.y, 2) for c in self.critters)
    positions.extend((s.x, s.y, 4 if s.warm else 3) for s in self.spikes)
    for x, y, channel in positions:
      state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), channel] = True
    values = (self.steps, self.held, round(self.swept * 50), self.hitstop,
              round((self.vx + 2) * 50), round((self.vy + 2) * 50))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
