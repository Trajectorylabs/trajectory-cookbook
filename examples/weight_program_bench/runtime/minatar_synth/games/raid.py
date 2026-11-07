from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class Bomb:
  x: float
  y: float
  vx: float
  vy: float


@dataclass
class Debris:
  x: float
  y: float
  vy: float


class RaidEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 15.0
    self.vx, self.vy, self.speed = 1.0, 0.0, 1.0
    self.down_distance = self.recoil = 0.0
    self.steps = self.frame = 0
    self.held = self.terminal = False
    self.bomb = None
    self.heights = self.random.uniform(10, 60, 8)
    self.debris = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    released = self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, released):
    scroll = max(0, self.y - 15) * 0.05
    if self.held:
      self.speed += 0.01
    else:
      self.speed += (1 - self.speed) * 0.2
      if released and self.bomb is None and self.frame > 30:
        self.bomb = Bomb(self.x, self.y, self.vx * self.speed, self.vy * self.speed)
        self.recoil = 0.33
    self.x += self.vx * self.speed
    self.y += self.vy * self.speed + self.recoil - scroll
    self.recoil *= 0.8
    if self.vy == 0:
      if (self.x < 10 and self.vx < 0) or (self.x > 90 and self.vx > 0):
        self.vy = 0.1
        self.down_distance = 1
    else:
      direction = 1 if self.x < 50 else -1
      self.vx += (direction - self.vx) * 0.05
      self.down_distance -= self.vy * self.speed
      if self.down_distance < 0:
        self.vx, self.vy = direction, 0
    bomb = self.bomb
    if bomb is not None:
      bomb.vy = (bomb.vy + 0.1) * 0.99
      bomb.vx *= 0.99
      bomb.x += bomb.vx
      bomb.y += bomb.vy - scroll
      if (bomb.x < 0 and bomb.vx < 0) or (bomb.x > 99 and bomb.vx > 0):
        bomb.vx *= -1
      if bomb.y > 99:
        self.bomb = bomb = None
    for fragment in self.debris:
      fragment.vy += 0.1
      fragment.y += fragment.vy - scroll
    self.debris = [fragment for fragment in self.debris if fragment.y <= 110]
    old_debris = tuple(self.debris)
    self.heights += scroll
    reward = 0
    for index, height in enumerate(self.heights):
      x = 10 + index * 10
      blocks = ceil(height / 10)
      base = 100 - height + blocks * 10
      for block in range(blocks):
        y = base - block * 10
        if bomb is not None and self._touches(x, y, bomb.x, bomb.y, 1.5, 1.5):
          removed = blocks - block
          reward += removed * (removed + 1) // 2
          self.heights[index] = max(0, 100 - y)
          self.bomb = bomb = None
          for offset in range(1, removed):
            self.debris.append(Debris(x, y - 10 * offset, -(offset + 1) * 0.5))
          break
        if any(abs(fragment.x - x) < 9 and abs(fragment.y - y) < 10 for fragment in old_debris):
          self.heights[index] = max(0, 100 - y)
        if self._touches(x, y, self.x, self.y, 3, 2.5):
          self.terminal = True
    self.frame += 1
    return reward

  @staticmethod
  def _touches(x, y, px, py, half_width, half_height):
    return x - half_width < px < x + 9 + half_width and y - half_height < py < y + 10 + half_height

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(np.clip(self.x // 10, 0, 9)), 0] = True
    if self.bomb is not None and 0 <= self.bomb.x < 100 and 0 <= self.bomb.y < 100:
      state[int(self.bomb.y // 10), int(self.bomb.x // 10), 1] = True
    rows = np.arange(10) * 10
    for index, height in enumerate(self.heights):
      if height > 0:
        state[rows + 10 > 110 - height, index + 1, 2] = True
    for fragment in self.debris:
      state[(rows < fragment.y + 10) & (rows + 10 > fragment.y), int(fragment.x // 10), 3] = True
    ahead = self.x + self.vx * 10
    state[int(np.clip((self.y + self.vy * 10) // 10, 0, 9)),
          int(np.clip(ahead // 10, 0, 9)), 4] = True
    for row, value in enumerate((self.steps, round(self.speed * 100), self.held, self.vy > 0)):
      for bit in range(11):
        state[row + bit // 10 * 4, bit % 10, 5] = bool(value & (1 << bit))
    return state
