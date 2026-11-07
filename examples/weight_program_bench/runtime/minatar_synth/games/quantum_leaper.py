from math import pi, sin

import numpy as np


LANES = (20, 37.5, 55, 72.5, 90)


class QuantumLeaperEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lane = self.gold = 2
    self.y = 50.0
    self.steps = self.frames = self.crossings = 0
    self.multiplier = 1
    self.gold_timer = 300
    self.held = self.terminal = False
    self.particles = []

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
    self.frames += 1
    phase = self.frames / 10
    displacement = sin(phase) * 10
    reward = 0
    if phase >= self.crossings * pi:
      self.crossings += 1
      if self.lane == self.gold:
        reward = self.multiplier
        self.multiplier = min(99, self.multiplier + 1)
      else:
        self.multiplier = max(1, self.multiplier - 1)
    self.y = LANES[self.lane] + displacement
    if tap:
      self.lane = (self.lane + (1 if displacement > 0 else -1)) % len(LANES)
    self.gold_timer -= 1
    if self.gold_timer < 0:
      self.gold = self.random.integers(len(LANES))
      self.gold_timer += 300
    self.particles = [(x - 1, lane) for x, lane in self.particles if x >= 1]
    self.terminal = any(abs(x - 20) < 3.5 and abs(LANES[lane] - self.y) < 3.5
                        for x, lane in self.particles)
    if self.random.random() < 0.02:
      self.particles.append((103, self.random.integers(len(LANES))))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y // 10)), 2, 0] = True
    for x, lane in self.particles:
      if x < 100:
        state[int(LANES[lane] // 10), int(x // 10), 1] = True
    state[int(LANES[self.gold] // 10), :, 2] = True
    state[int(LANES[self.lane] // 10), 2, 3] = True
    next_lane = (self.lane + (1 if sin((self.frames + 1) / 10) > 0 else -1)) % len(LANES)
    state[int(LANES[next_lane] // 10), 2, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.gold_timer, self.held)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
