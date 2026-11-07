from math import hypot, pi, sin

import numpy as np


class PressMEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(400, horizon)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 50
    self.steps = self.action = self.frame = 0
    self.multiplier = 1
    self.terminal = False
    self.ends = np.repeat([[40.0], [60.0]], 10, axis=1)
    self.walls = self.ends.copy()
    self.flying = []
    self._new_cycle()

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _new_cycle(self):
    self.multiplier = max(1, self.multiplier - 1)
    self.starts = np.empty((2, 10))
    for side, center in enumerate((20, 80)):
      angle = self.random.uniform(0, 2 * pi)
      velocity = self.random.uniform(0.4, 1) * self.random.choice((-1, 1))
      radius = self.random.uniform(10, 20)
      y = center + self.random.uniform(-5, 5)
      for col in range(10):
        self.starts[side, col] = np.clip(y + sin(angle) * radius, 2, 97)
        velocity += self.random.uniform(-0.1, 0.1)
        radius += self.random.uniform(-1, 1)
        angle += velocity
        y += self.random.uniform(-1, 1)
    gaps = self.starts[1] - self.starts[0]
    narrowest = np.argmin(gaps)
    self.targets = self.starts + np.array([[1], [-1]]) * gaps[narrowest] / 2
    if not np.any((self.targets[0] < 40) & (self.targets[1] > 60)):
      for col in range(10):
        if col != narrowest:
          adjustment = np.array([min(0, 40 - self.targets[0, col]),
                                 max(0, 60 - self.targets[1, col])])
          self.starts[:, col] += adjustment
          self.targets[:, col] += adjustment
    self.attached = list(zip(*np.where(self.random.random((2, 10)) < 0.2)))

  def act(self, action):
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

  def _advance(self):
    self.x = min(96, max(3, self.x + 3 * ((self.action == 3) - (self.action == 1))))
    if self.frame < 60:
      self.walls = self.ends + (self.starts - self.ends) * (self.frame / 60)
    else:
      remaining = 79 - self.frame
      amount = 1 - remaining / 5 if remaining < 5 else (1 - remaining / 20) * 0.2
      self.walls = self.starts + (self.targets - self.starts) * amount
      if remaining < 5:
        self.terminal = any(
          col * 10 + 9 > self.x - 3 and col * 10 < self.x + 3
          and (self.walls[0, col] > 47.5 or self.walls[1, col] < 52.5)
          for col in range(10)
        )
    attached = []
    for side, col in self.attached:
      x = col * 10 + 5
      y = self.walls[side, col] + (4 if side == 0 else -4)
      if self.frame >= 60 and hypot(self.x - x, 50 - y) < 30:
        self.flying.append((x, y, self.random.uniform(0.1, 0.3)))
      else:
        attached.append((side, col))
    self.attached = attached
    reward = 0
    flying = []
    for x, y, speed in self.flying:
      x += (self.x - x) * speed
      y += (50 - y) * speed
      if abs(x - self.x) < 5 and abs(y - 50) < 5:
        reward += self.multiplier
        self.multiplier += 1
      else:
        flying.append((x, y, speed))
    self.flying = flying
    self.frame += 1
    if self.frame == 80:
      self.frame = 0
      self.ends = self.targets
      self._new_cycle()
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, int(self.x // 10), 0] = True
    rows = np.arange(10)[:, None] * 10
    state[:, :, 1] = rows < self.walls[0]
    state[:, :, 2] = rows + 10 > self.walls[1]
    for side, col in self.attached:
      y = self.walls[side, col] + (4 if side == 0 else -4)
      state[int(np.clip(y // 10, 0, 9)), col, 3] = True
    for x, y, _ in self.flying:
      state[int(np.clip(y // 10, 0, 9)), int(x // 10), 4] = True
    for row, value in enumerate((self.frame, self.multiplier, self.steps, self.action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
