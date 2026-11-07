from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Laser:
  x: float
  y: float
  vy: int
  width: float
  burst: int = 0


@dataclass
class Fragment:
  x: float
  y: float
  vy: int
  height: float


class LRainEnvironment:
  score_cap = 3000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x = self.safe_x = 50.0
    self.safe_vx = self.invincible = self.timer = 0.0
    self.energy = self.steps = self.frames = self.action = 0
    self.multiplier = 1
    self.terminal = False
    self.lasers: list[Laser] = []
    self.fragments: list[Fragment] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
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

  def _advance(self) -> int:
    previous = self.x
    self.x = min(99, max(0, self.x + 2 * ((self.action == 3) - (self.action == 1))))
    self.safe_x += self.safe_vx / (len(self.lasers) + 1)
    self.safe_vx += (self.safe_vx + self.random.uniform(-0.01, 0.01) + 0.1) % 0.2 - 0.1
    self.safe_vx *= 0.99
    if (self.safe_x < 9 and self.safe_vx < 0) or (self.safe_x > 90 and self.safe_vx > 0):
      self.safe_vx *= -1
    self.frames += 1
    self.timer -= 1
    if self.timer < 0:
      x, width = self.random.uniform(0, 99), self.random.uniform(3, 9)
      vy = 3 if self.random.integers(2) else -3
      if abs(x - self.safe_x) > width + 2:
        self.lasers.append(Laser(x, 0 if vy > 0 else 99, vy, width))
      cadence = abs(math.cos(self.frames * 0.001))
      self.timer = self.random.uniform(0, 30 / (5 if cadence < 0.15 else cadence))
    self._move_lasers()
    reward = 0
    hit = False
    remaining = []
    for fragment in self.fragments:
      fragment.y += fragment.vy
      if abs(fragment.y - 50) < fragment.height / 2 + 3:
        left, right = fragment.x - 1.5, fragment.x + 1.5
        if ((right > self.x + 2 and left < previous + 6)
            or (right > previous - 6 and left < self.x - 2)):
          reward += self.multiplier
          self.energy += 1
          if self.energy > 150:
            self.multiplier += 1
            self.energy = 0
            self.invincible = 30
        hit |= abs(fragment.y - 50) < fragment.height / 2 + 2 and abs(fragment.x - self.x) < 2
      if -fragment.height / 2 <= fragment.y <= 99 + fragment.height / 2:
        remaining.append(fragment)
    self.fragments = remaining
    self.terminal = hit and self.invincible < 3
    if self.invincible >= 3:
      self.invincible *= 0.98
    return reward

  def _move_lasers(self) -> None:
    remaining = []
    for laser in self.lasers:
      if 0 <= laser.y <= 99:
        laser.y += laser.vy
      else:
        laser.burst += 1
        width = min(math.sin(laser.burst * 0.1) * laser.width * 2, laser.width - 1)
        if width <= 0:
          continue
        height = self.random.uniform(9, 15)
        for _ in range(int(width)):
          x = laser.x + self.random.uniform(-width, width)
          offset = self.random.uniform(0, 9) + height / 2
          y = 100 - laser.y - math.copysign(offset, laser.vy)
          self.fragments.append(Fragment(x, y, laser.vy * 3, height))
      remaining.append(laser)
    self.lasers = remaining

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, int(self.x) // 10, 0] = True
    for laser in self.lasers:
      left = max(0, int((laser.x - laser.width) // 10))
      right = min(9, int((laser.x + laser.width) // 10))
      head = min(9, max(0, int(laser.y // 10)))
      channel = 1 if laser.vy > 0 else 2
      rows = slice(0, head + 1) if laser.vy > 0 else slice(head, 10)
      state[rows, left:right + 1, channel] = True
    for fragment in self.fragments:
      if 0 <= fragment.x < 100:
        top = max(0, int((fragment.y - fragment.height / 2) // 10))
        bottom = min(9, int((fragment.y + fragment.height / 2) // 10))
        if top <= bottom:
          state[top:bottom + 1, int(fragment.x) // 10, 3 if fragment.vy > 0 else 4] = True
    gauges = (self.steps, round(self.x * 4), self.energy, min(511, self.multiplier),
              round(self.invincible * 10))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
