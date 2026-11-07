from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass
class Wave:
  lane: int
  amplitude: float
  x: float = -5


@dataclass
class Bubble:
  lane: int
  y: float
  vx: float = -0.2
  x: float = 98
  vy: float = 0


class PulseStreamEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.steps = self.frame = self.last_tap = self.auto_timer = self.spawn_lane = 0
    self.rapid_rate = 0.0
    self.held = self.terminal = False
    self.disabled = [False] * 5
    self.recovery = [0] * 5
    self.waves = []
    self.bubbles = [Bubble(2, self.random.uniform(-2, 2))]
    self.sequence = []

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

  def _spawn(self):
    if self.frame % 36 == 0:
      if len(self.sequence) <= 1:
        self.sequence = self.random.permutation(np.tile(np.arange(5), 3)).tolist()
      lane = self.sequence.pop(0)
      for offset in range(5):
        lane = (lane + offset) % 5
        if not self.disabled[lane]:
          self.waves.append(Wave(lane, self.random.uniform(1, 1.7)))
          break
    if self.random.random() < 0.01:
      if self.random.random() < 0.7:
        lane = self.spawn_lane
        self.spawn_lane = (self.spawn_lane + 1) % 5
      else:
        lane = self.random.integers(5)
      while self.disabled[lane]:
        lane = (lane + 1) % 5
      self.bubbles.append(Bubble(lane, self.random.uniform(-2, 2),
                                 self.random.uniform(-0.3, -0.2)))

  def _advance(self, tap):
    self._spawn()
    if tap:
      self.last_tap = self.frame
      self.auto_timer = 0
    elif self.frame - self.last_tap >= 600:
      self.auto_timer += 1
      if self.auto_timer == 60:
        tap = True
        self.auto_timer = 0
    if tap:
      self.rapid_rate += 0.3
      for lane, disabled in enumerate(self.disabled):
        if disabled:
          self.recovery[lane] += 1
          if self.recovery[lane] == 36:
            self.disabled[lane] = False
            self.recovery[lane] = 0
    self.rapid_rate *= 0.98
    for wave in self.waves:
      wave.x += 0.8
    self.waves = [wave for wave in self.waves if wave.x <= 105]
    reward = 0
    remaining = []
    for bubble in self.bubbles:
      if self.disabled[bubble.lane]:
        continue
      bubble.x += bubble.vx
      bubble.y += bubble.vy
      if tap:
        hits = 0
        for wave in self.waves:
          if wave.lane == bubble.lane and hypot(wave.x - bubble.x, bubble.y) < wave.amplitude * 6:
            hits += 1
            bubble.vx += 0.8
            bubble.vy += self.random.uniform(-0.2, 0.2)
        reward += hits
        if not hits:
          bubble.vx -= 0.1 * (0.25 + self.rapid_rate)
      bubble.vx *= 0.993
      bubble.vy *= 0.891
      bubble.y *= 0.95
      if bubble.x < 0:
        self.disabled[bubble.lane] = True
        self.recovery[bubble.lane] = 0
      elif bubble.x > 100:
        reward += 2
      else:
        remaining.append(bubble)
    self.bubbles = [bubble for bubble in remaining if not self.disabled[bubble.lane]]
    self.waves = [wave for wave in self.waves if not self.disabled[wave.lane]]
    self.terminal = sum(self.disabled) >= 3
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for wave in self.waves:
      columns = np.arange(5, 100, 10)
      state[2 * wave.lane, :, 0] |= abs(columns - wave.x) < wave.amplitude * 6 + 5
    for bubble in self.bubbles:
      col = min(9, int(bubble.x // 10))
      state[2 * bubble.lane, col, 1 + int(bubble.vx > 0)] = True
    for lane, disabled in enumerate(self.disabled):
      state[2 * lane, :, 3] = disabled
      for bit in range(6):
        state[2 * lane, bit, 4] = bool(self.recovery[lane] & (1 << bit))
    for row, value in enumerate((self.steps, self.frame % 36, self.held,
                                 min(1023, self.frame - self.last_tap), self.auto_timer,
                                 round(self.rapid_rate * 100))):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
