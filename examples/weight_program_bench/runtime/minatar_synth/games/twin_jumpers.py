from dataclasses import dataclass

import numpy as np


@dataclass
class Jumper:
  x: float
  y: float = 70.
  vx: int = 1
  vy: float = 0.
  grace: int = 0


@dataclass
class Platform:
  x: float
  y: float
  width: float


class TwinJumpersEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.jumpers = [Jumper(20), Jumper(80)]
    self.platforms = [Platform(50, 80, 100)]
    for y in (70, 40, 10):
      for low in (0, 60):
        self.platforms.append(Platform(self.random.uniform(low, low + 40), y,
                                       self.random.uniform(20, 50)))
    self.power = 1.
    self.platform_distance = 0.
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
    jump = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0.
    for frame in range(4):
      reward += self._advance(jump and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self, jump):
    scroll = 0.1 + max(0, 60 - max(jumper.y for jumper in self.jumpers)) * 0.05
    if jump:
      for jumper, strength in zip(self.jumpers, (3, 2.5), strict=True):
        if jumper.grace > 0:
          jumper.vy = -strength * self.power
      self.power *= 0.8
    else:
      self.power = min(1, self.power + 0.01)

    previous_y = [jumper.y for jumper in self.jumpers]
    for jumper in self.jumpers:
      jumper.y += jumper.vy
      jumper.vy += 0.1 if self.action == 5 else 0.2
      jumper.x += jumper.vx * 0.6
      jumper.grace = max(0, jumper.grace - 1)
      if (jumper.x < 3 and jumper.vx < 0) or (jumper.x > 97 and jumper.vx > 0):
        jumper.vx *= -1
      if jumper.y < 0 and jumper.vy < 0:
        jumper.y = 0
        jumper.vy *= -0.5

    for platform in self.platforms:
      platform.y += scroll
    self.platforms = [platform for platform in self.platforms if platform.y <= 109]
    self.platform_distance -= scroll
    if self.platform_distance <= 0:
      for low in (0, 50):
        self.platforms.append(Platform(self.random.uniform(low, low + 50),
                                       -self.random.uniform(5, 25),
                                       self.random.uniform(30, 50)))
      self.platform_distance = self.random.uniform(20, 30)
    for jumper, old_y, offset in zip(self.jumpers, previous_y, (4, 3), strict=True):
      if jumper.vy <= 0:
        continue
      supports = [
        platform.y - offset for platform in self.platforms
        if abs(platform.x - jumper.x) < platform.width / 2 + 2
        and min(jumper.y - 2, old_y - scroll) <= platform.y - offset <= jumper.y + 1
      ]
      if supports:
        jumper.y = min(supports)
        jumper.vy = 0.
        jumper.grace = 9
    self.terminal = any(jumper.y > 102 for jumper in self.jumpers)
    return scroll

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for index, jumper in enumerate(self.jumpers):
      row = int(np.clip(jumper.y // 10, 0, 9))
      col = int(jumper.x // 10)
      state[row, col, index] = True
      state[row, col, 3 + index] = jumper.grace > 0
    cols = np.arange(5, 100, 10)
    for platform in self.platforms:
      if 0 <= platform.y < 100:
        state[int(platform.y // 10), :, 2] |= abs(cols - platform.x) < platform.width / 2 + 5
    values = (self.steps, self.action, round(self.power * 100), round(self.platform_distance * 10),
              *(value for jumper in self.jumpers for value in
                (jumper.vx > 0, jumper.grace, round((jumper.vy + 4) * 32))))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
