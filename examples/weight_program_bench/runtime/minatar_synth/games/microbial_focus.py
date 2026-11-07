from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Microbe:
  x: float
  y: float
  depth: float
  size: float
  vx: float = 0
  vy: float = 0
  age: int = 0
  heading: float | None = None


class MicrobialFocusEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.focus = 0.5
    self.direction = 1
    self.frames = self.steps = 0
    self.next_spawn = 0.0
    self.multiplier = 1.0
    self.pressed = self.terminal = False
    self.microbes: list[Microbe] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction *= -1
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
    for _ in range(8):
      depth = self.random.uniform(0.05, 0.95)
      if abs(depth - self.focus) > 0.15:
        break
    edge = self.random.integers(4)
    size = self.random.uniform(4, 8)
    along = self.random.uniform(10, 90)
    drift = self.random.uniform(-0.2, 0.2)
    speed = self.random.uniform(0.5, 1)
    outside = size - 1
    x, y, vx, vy = ((along, -outside, drift, speed),
                    (100 + outside, along, -speed, drift),
                    (along, 100 + outside, drift, -speed),
                    (-outside, along, speed, drift))[edge]
    self.microbes.append(Microbe(x, y, depth, size, vx, vy))

  def _advance(self):
    if self.pressed:
      self.focus += self.direction * 0.015
      if not 0 <= self.focus <= 1:
        self.focus = min(1, max(0, self.focus))
        self.multiplier = 1
    self.multiplier *= 0.999
    if self.frames >= self.next_spawn and len(self.microbes) < 24:
      self.next_spawn = self.frames + 50 * math.sqrt(len(self.microbes) + 1)
      self._spawn()
    self.frames += 1
    reward = 0
    remaining = []
    for microbe in self.microbes:
      microbe.age += 1
      if abs(microbe.depth - self.focus) < 0.1:
        if microbe.heading is None:
          dx, dy = microbe.x - 50, microbe.y - 50
          microbe.heading = math.atan2(dy, dx) if dx or dy else self.random.uniform(0, math.tau)
        microbe.vx += math.cos(microbe.heading) * 0.02
        microbe.vy += math.sin(microbe.heading) * 0.02
        microbe.x += microbe.vx
        microbe.y += microbe.vy
      else:
        microbe.vx *= 0.99
        microbe.vy *= 0.99
        microbe.x += microbe.vx
        microbe.y += microbe.vy
        if abs(microbe.vx) < 0.01 and abs(microbe.vy) < 0.01:
          microbe.heading = None
          microbe.vx = microbe.vy = 0
      low, high = -microbe.size, 100 + microbe.size
      if low <= microbe.x <= high and low <= microbe.y <= high:
        remaining.append(microbe)
        self.terminal |= microbe.age >= 600
      else:
        reward += math.floor(self.multiplier + 0.5)
        self.multiplier += 1
    self.microbes = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for microbe in self.microbes:
      row = min(9, max(0, int(microbe.y // 10)))
      col = min(9, max(0, int(microbe.x // 10)))
      state[row, col, int(microbe.depth * 4)] = True
      state[row, col, 4] |= microbe.age >= 360
    oldest = max(self.microbes, key=lambda m: m.age, default=None)
    gauges = (self.steps, round(self.focus * 500), round(self.multiplier * 10),
              oldest.age // 2 if oldest else 0, round(oldest.depth * 500) if oldest else 0)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, value) & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
