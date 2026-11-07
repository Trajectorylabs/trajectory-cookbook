from dataclasses import dataclass

import numpy as np


@dataclass
class Pillar:
  x: float
  depth: float
  width: float
  height: float
  bonus: bool = False

  def bounds(self):
    x = 50 + self.x / self.depth
    half_width = self.width / (2 * self.depth)
    return (x - half_width, x + half_width,
            60 - self.height / (6 * self.depth), 60 + 5 * self.height / (6 * self.depth))


class Pillars3dEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vy = 50.0, 10.0, 0.0
    self.pillars = [Pillar(0, 20, 100, 100, True)]
    self.timer = self.bonus_timer = 9
    self.steps = self.action = 0
    self.multiplier = 1
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

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
    self.timer -= 1
    if self.timer < 0:
      self.bonus_timer -= 1
      bonus = self.bonus_timer < 0
      x = self.random.uniform(60, 160) * self.random.choice((-1, 1))
      self.pillars.append(Pillar(x, 20, self.random.uniform(50, 100),
                                 self.random.uniform(70, 180), bonus))
      self.timer = 20
      if bonus:
        self.bonus_timer = 9
    self.x = min(93, max(6, self.x + 1.5 * ((self.action == 3) - (self.action == 1))))
    self.y += self.vy
    self.vy += 0.1
    if self.y > 95:
      self.terminal = True
      return 0
    reward = 0
    remaining = []
    collision_y = self.y
    for pillar in self.pillars:
      left, right, top, bottom = pillar.bounds()
      if left < self.x + 3 and right > self.x - 5 and top < collision_y + 3 and bottom > collision_y - 3:
        if self.vy > 0:
          self.vy = -2.5
          self.y = min(self.y, top)
        reward += self.multiplier
        self.multiplier += int(pillar.bonus)
      else:
        pillar.depth -= 0.2
        if pillar.depth > 1:
          remaining.append(pillar)
    self.pillars = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    col = int(self.x // 10)
    state[min(9, max(0, int(self.y // 10))), col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for pillar in self.pillars:
      left, right, top, bottom = pillar.bounds()
      visible = ((cols + 5 > left) & (cols - 5 < right) &
                 (rows + 5 > top) & (rows - 5 < bottom))
      state[:, :, 2 if pillar.bonus else 1] |= visible
      state[:, :, 3] |= visible & (pillar.depth < 6)
    state[min(9, max(0, int((self.y + 4 * self.vy) // 10))), col, 4] = True
    values = (self.steps, self.multiplier, round(self.x * 4), round((self.vy + 4) * 40), self.action)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    return state
