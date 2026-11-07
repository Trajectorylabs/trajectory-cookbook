from dataclasses import dataclass

import numpy as np


@dataclass
class Hazard:
  x: float
  y: float
  speed: float


class MonkeyTEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vx, self.vy = 50.0, 90.0, 0.3, 0.0
    self.count = self.compression = self.power = self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self._spawn_coins()
    self.hazards = [Hazard(self.random.uniform(10, 90), self.random.uniform(10, 70),
                           0.5 + i * 0.1) for i in range(3)]

  def _spawn_coins(self):
    self.coins = [(self.random.uniform(30, 70), self.random.uniform(20, 70)) for _ in range(5)]

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.power = self.compression = 0
    elif self.pressed and not pressed:
      self.power = self.compression
      self.compression = self.count = 0
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  @property
  def trampoline_y(self):
    return 90 + (self.compression + self.power) * 0.25

  def _advance(self):
    if self.pressed:
      self.compression = min(30, self.compression + 1)
    self.vy += 0.1
    self.x += self.vx
    self.y += self.vy
    if (self.x < 25 and self.vx < 0) or (self.x > 75 and self.vx > 0):
      self.vx *= -1
    if self.vy > 0 and abs(self.y - self.trampoline_y) < 4:
      self.y = self.trampoline_y - 5
      self.vy = -self.vy * 0.2 - self.power * 0.13
      if self.power > 1:
        self.power = 0
    if self.y > 99:
      self.y = 80
    remaining = []
    reward = 0
    for x, y in self.coins:
      if abs(x - self.x) < 5 and abs(y - self.y) < 5:
        self.multiplier += self.count
        self.count += 1
        reward += self.multiplier
      else:
        remaining.append((x, y))
    self.coins = remaining
    if not self.coins:
      self._spawn_coins()
    for hazard in self.hazards:
      inward = (hazard.x < 10 and hazard.speed > 0) or (hazard.x > 90 and hazard.speed < 0)
      hazard.x += hazard.speed * (0.5 if inward else 1)
      if (hazard.x < -10 and hazard.speed < 0) or (hazard.x > 110 and hazard.speed > 0):
        hazard.y = self.random.uniform(10, 70)
        hazard.speed *= -1
      self.terminal |= abs(hazard.x - self.x) < 5 and abs(hazard.y - self.y) < 5
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), int(self.x // 10), 0] = True
    for x, y in self.coins:
      state[int(y // 10), int(x // 10), 1] = True
    for hazard in self.hazards:
      if 0 <= hazard.x < 100:
        state[int(hazard.y // 10), int(hazard.x // 10), 2 + (hazard.speed < 0)] = True
    state[int(self.trampoline_y // 10), 2:8, 4] = True
    values = (self.steps, self.compression, self.power, self.count, self.multiplier,
              round((self.vy + 10) * 20))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, :2, 5] = self.pressed, self.vx > 0
    return state
