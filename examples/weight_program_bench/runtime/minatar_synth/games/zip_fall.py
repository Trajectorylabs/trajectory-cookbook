from dataclasses import dataclass

import numpy as np


@dataclass
class Line:
  y: float


@dataclass
class Object:
  x: float
  y: float
  vx: float
  coin: bool


class ZipFallEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lines = [Line(y) for y in (20, 50, 75)]
    self.attached = self.lines[0]
    self.x, self.y, self.vx, self.vy = 20., 20., 1, 0.
    self.objects = []
    self.line_distance = self.object_timer = 0.
    self.multiplier = 1
    self.steps = self.action = 0
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
    drop = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(drop and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, drop):
    scroll = 0.05 + max(0, self.y - 30) * 0.1
    if drop:
      self.y += 5
      self.vy = 1
      self.attached = None
    if self.attached is None:
      self.vy += 0.1
      self.y += self.vy - scroll
    else:
      self.y = self.attached.y + 3
      self.x += self.vx
    if (self.x < 20 and self.vx < 0) or (self.x > 80 and self.vx > 0):
      self.vx *= -1
      self.multiplier = max(1, self.multiplier - 1)
    if self.y < 3:
      self.terminal = True
      return 0

    for line in self.lines:
      if self.attached is None and 17 < self.x < 83 and abs(line.y - self.y) < 4:
        self.attached = line
      line.y -= scroll
    self.lines = [line for line in self.lines if line.y >= 0]
    self.line_distance -= scroll
    if self.line_distance < 0:
      self.lines.append(Line(100))
      self.line_distance = self.random.uniform(20, 40)

    self.object_timer -= 1
    if self.object_timer <= 0:
      coin = self.random.random() >= 0.6
      vx = self.random.uniform(0.5, 1) * self.random.choice((-1, 1))
      self.objects.append(Object(-3 if vx > 0 else 103, self.random.uniform(0, 120), vx, coin))
      self.object_timer = self.random.uniform(20, 30)
    reward = 0
    remaining = []
    for obj in self.objects:
      obj.y -= scroll
      obj.x += obj.vx
      if abs(obj.x - self.x) < 5 and abs(obj.y - self.y) < 5:
        if obj.coin:
          reward += self.multiplier
          self.multiplier += 3
          continue
        self.terminal = True
      if -5 <= obj.x <= 105:
        remaining.append(obj)
    self.objects = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = int(np.clip(self.y // 10, 0, 9)), int(self.x // 10)
    state[row, col, 0] = True
    state[row, col, 4] = self.attached is not None
    for line in self.lines:
      state[min(9, int(line.y // 10)), 2:8, 1] = True
    for obj in self.objects:
      if 0 <= obj.x < 100 and 0 <= obj.y < 100:
        state[int(obj.y // 10), int(obj.x // 10), 3 if obj.coin else 2] = True
    values = (self.steps, self.action, self.vx > 0, round(self.vy * 32), self.multiplier,
              round(self.line_distance * 8), round(self.object_timer * 8))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
