from dataclasses import dataclass

import numpy as np


@dataclass
class Floor:
  x: float
  y: float
  width: float


class MirrorFloorEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.floors = [Floor(10, 50, 80)]
    self.coins = [(60, 47)]
    self.distance = 0.0
    self.y, self.mirror_y, self.vy, self.speed = 10.0, 0.0, 0.0, 1.0
    self.side = self.multiplier = 1
    self.steps = 0
    self.airborne = True
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    scroll = 0.5 * self.speed
    if tap and not self.airborne:
      self.vy = -1.5 * self.side
      self.airborne = True
    if self.airborne:
      self.vy += 0.07 * self.side
      self.y += self.vy
    self.speed += (2 - self.speed) * 0.05 if self.pressed else (1 - self.speed) * 0.2
    self.terminal = (self.y < 0 and self.side == -1) or (self.y > 99 and self.side == 1)
    if self.terminal:
      return 0
    self.distance -= scroll
    if self.distance < 0:
      floor = Floor(100, self.random.uniform(10, 90), self.random.uniform(45, 75))
      self.floors.append(floor)
      x = self.random.uniform(20, 25)
      while x < floor.width - 20:
        self.coins.append((100 + x, floor.y - 3))
        x += self.random.uniform(15, 30)
      self.distance += floor.width + self.random.uniform(10, 20)
    supported = False
    for floor in self.floors:
      floor.x -= scroll
      if floor.x - 3 < 9 < floor.x + floor.width + 3:
        supported = True
        if floor.y - 3 < self.y < floor.y + 5 and self.vy * self.side > 0:
          self.airborne = False
          self.y = floor.y + (-3 if self.side == 1 else 5)
        self.mirror_y = 2 * floor.y - self.y + 2
    self.floors = [floor for floor in self.floors if floor.x + floor.width >= 0]
    if not supported:
      self.airborne = True
    reward = 0
    remaining = []
    for x, y in self.coins:
      x -= scroll
      if abs(x - 9) < 5 and (abs(y - self.y) < 5 or (supported and abs(y - self.mirror_y) < 5)):
        reward += self.multiplier
        self.multiplier += 1
        self.side *= -1
        self.vy *= -1
        self.y = self.mirror_y
      elif x < -3:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append((x, y))
    self.coins = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for y, channel in ((self.y, 0), (self.mirror_y, 1)):
      if 0 <= y < 100:
        state[int(y // 10), 0, channel] = True
    for floor in self.floors:
      left = max(0, int(floor.x // 10))
      right = min(9, int((floor.x + floor.width) // 10))
      if left <= right:
        state[int(floor.y // 10), left:right + 1, 2] = True
    for x, y in self.coins:
      if 0 <= x < 100:
        state[int(y // 10), int(x // 10), 3] = True
        state[int((y + 8) // 10), int(x // 10), 4] = True
    values = (self.steps, min(511, self.multiplier), round(self.speed * 100),
              min(511, max(0, round((self.vy + 6) * 40))))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 5] = self.pressed, self.airborne, self.side == 1
    return state
