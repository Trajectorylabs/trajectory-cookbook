from dataclasses import dataclass

import numpy as np


@dataclass
class Barrel:
  x: float
  speed: float
  scored: bool = False


class TarutobiEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = self.cursor = 30.
    self.vx = self.vy = 0.
    self.multiplier = 1
    self.steps = self.action = 0
    self.airborne = True
    self.terminal = False
    self.barrels = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
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

  def _advance(self):
    self.cursor = min(120, max(0, self.cursor + 3 * ((self.action == 3) - (self.action == 1))))
    if self.random.random() < 0.01:
      self.barrels.append(Barrel(123, self.random.uniform(0.4, 0.8)))
    reward = 0
    dangerous = self.airborne
    for barrel in self.barrels:
      barrel.x -= barrel.speed
      if not barrel.scored and barrel.x < self.x:
        reward += self.multiplier
        self.multiplier += 1
        barrel.scored = True
    self.barrels = [barrel for barrel in self.barrels if barrel.x > 0]
    if self.airborne:
      self.vy += 0.1
      if self.y > 47:
        self.airborne = False
        self.y, self.vy = 47, 0.
    else:
      self.vx *= 0.95
      self.vy *= 0.95
      self.multiplier = 1
    if not 0 <= self.x <= 120:
      self.x = min(120, max(0, self.x))
      self.vx = 0.
    offset = min(7, max(-7, self.cursor - self.x))
    self.vx += offset * (0.004 if self.airborne else 0.01)
    self.x += self.vx
    self.y += self.vy
    if any(abs(barrel.x - self.x) < 5.5 for barrel in self.barrels) and abs(47 - self.y) < 5.5:
      if dangerous:
        self.terminal = True
      else:
        self.airborne = True
        self.y, self.vy = 42, -1.2 - abs(self.vx)
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(np.clip(self.y // 6, 0, 9))
    col = int(np.clip(self.x // 12, 0, 9))
    state[row, col, 0] = True
    for barrel in self.barrels:
      if 0 <= barrel.x < 120:
        state[7, int(barrel.x // 12), 2 if self.airborne else 1] = True
    state[8:, :, 3] = True
    state[9, min(9, int(self.cursor // 12)), 4] = True
    values = (self.steps, self.airborne, self.multiplier, self.action,
              round((self.vx + 4) * 50), round((self.vy + 4) * 50))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(int(np.clip(value, 0, 511)) & (1 << bit))
    return state
