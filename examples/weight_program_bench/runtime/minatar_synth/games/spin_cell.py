from math import floor, hypot, sqrt

import numpy as np


CORNERS = ((14, 14), (86, 14), (86, 86), (14, 86))


class SpinCellEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 40.
    self.vx, self.vy = self.random.uniform(0.4, 0.8), 0.
    self.spikes = {1, 7}
    self.gem = int(self.random.integers(4))
    self.timer, self.warning = 400, -1
    self.steps = self.death = 0
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
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
    if self.death:
      self.death -= 1
      self.terminal = self.death == 0
      return 0
    if tap:
      self.spikes = {(segment + 1) % 12 for segment in self.spikes}
      if self.warning >= 0:
        self.warning = (self.warning + 1) % 12
      if self.y > 87:
        self.vx = max(-1.4, min(1.4, self.vx - 0.6))
    self.timer -= 1
    if self.timer < 0:
      if self.warning >= 0:
        self.spikes.add(self.warning)
        self.warning = -1
        self.timer = 200 * sqrt(len(self.spikes) + 1)
      elif len(self.spikes) < 8:
        self.warning = int(self.random.choice(sorted(set(range(12)) - self.spikes)))
        self.timer = 90
      else:
        self.timer = 500
    self.vy += 0.06
    self.x += self.vx
    self.y += self.vy
    segment = self._bounce()
    if segment in self.spikes:
      self.death = 7
      return 0
    x, y = CORNERS[self.gem]
    if hypot(self.x - x, self.y - y) < 8:
      reward = max(1, len(self.spikes)) ** 2
      if self.spikes:
        self.spikes.remove(int(self.random.choice(sorted(self.spikes))))
      self.gem = (self.gem + int(self.random.integers(1, 4))) % 4
      return reward
    return 0

  def _bounce(self):
    if self.y < 9:
      side, offset = 0, self.x
      self.y, self.vy = 9, abs(self.vy) * 0.9
    elif self.x > 91:
      side, offset = 1, self.y
      self.x, self.vx = 91, -abs(self.vx) * 0.95
    elif self.y > 91:
      side, offset = 2, self.x
      self.y, self.vy = 91, -3.1
      self.vx = min(1.2, max(-1.2, self.vx + self.random.uniform(-0.25, 0.25)))
    elif self.x < 9:
      side, offset = 3, self.y
      self.x, self.vx = 9, abs(self.vx) * 0.95
    else:
      return -1
    index = min(2, max(0, floor((offset - 7) * 3 / 86)))
    return side * 3 + (2 - index if side >= 2 else index)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), int(self.x // 10), 0] = True
    gx, gy = CORNERS[self.gem]
    state[gy // 10, gx // 10, 1] = True
    for side in range(4):
      for index in range(3):
        segment = side * 3 + index
        channel = 2 if segment in self.spikes else 3 if segment == self.warning else 4
        span = slice(1 + 3 * index, min(9, 4 + 3 * index))
        reverse = slice(10 - span.stop, 10 - span.start)
        if side == 0:
          state[0, span, channel] = True
        elif side == 1:
          state[span, 9, channel] = True
        elif side == 2:
          state[9, reverse, channel] = True
        else:
          state[reverse, 0, channel] = True
    values = (self.steps, self.held, self.death, round(self.timer),
              round((self.vx + 2) * 100), round((self.vy + 8) * 40))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
