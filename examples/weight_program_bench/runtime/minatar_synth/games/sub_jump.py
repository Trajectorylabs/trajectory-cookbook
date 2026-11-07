from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
  x: float
  y: float


class SubJumpEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 5., 60.
    self.vx = self.vy = self.terrain_velocity = 0.
    self.terrain_distance, self.coin_distance = 50., 0.
    self.steps = 0
    self.multiplier = 1
    self.land = self.held = self.terminal = False
    self.points = [Point(i * 10 - 10, self.random.uniform(75, 85)) for i in range(12)]
    self.coins = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _scroll_terrain(self, scroll):
    self.terrain_distance -= scroll
    for index, point in enumerate(self.points):
      point.x -= scroll
      if point.x >= -10:
        continue
      if self.terrain_distance < 0:
        self.land = not self.land
        self.terrain_distance = self.random.uniform(200, 300) / (7 if self.land else 1)
      point.x += 120
      previous = self.points[index - 1].y
      self.terrain_velocity = (self.terrain_velocity + self.random.uniform(-2, 2)) * 0.95
      if self.land:
        if previous > 50:
          self.terrain_velocity -= 5
        elif (previous < 40 and self.terrain_velocity < 0) or (previous > 45 and self.terrain_velocity > 0):
          self.terrain_velocity *= -0.5
        y = previous + self.terrain_velocity / 3
      else:
        if previous < 65:
          self.terrain_velocity += 5 if previous < 55 else 3
        elif previous > 90 and self.terrain_velocity > 0:
          self.terrain_velocity *= -0.5
        y = previous + self.terrain_velocity
        if self.terrain_distance < 60:
          self.terrain_velocity += 4
      point.y = min(95, max(35, y)) + self.random.uniform(-5, 5)

  def _advance(self):
    scroll = max(0, self.x - 10) * 0.2
    self._scroll_terrain(scroll)
    submerged = self.y > 50
    if self.held:
      self.vy += -0.06 if submerged else 0.01
    else:
      self.vy += 0.03 if submerged else 0.05
    self.vx += ((1 if self.held else 0.5) - self.vx) * 0.1
    drag = 0.95 if submerged else 0.99
    self.vx *= drag
    self.vy *= drag
    self.x += self.vx - scroll
    self.y += self.vy
    ordered = sorted(self.points, key=lambda point: point.x)
    terrain_y = np.interp(self.x, [point.x for point in ordered], [point.y for point in ordered])
    self.terminal = bool(abs(self.y - terrain_y) < 4)
    self.coin_distance -= scroll
    if self.coin_distance < 0:
      edge = ordered[-1]
      offset = self.random.uniform(5, 10) if edge.y < 50 else self.random.uniform(10, 20)
      self.coins.append(Point(edge.x, edge.y - offset))
      self.coin_distance = self.random.uniform(10, 40)
    reward = 0
    remaining = []
    for coin in self.coins:
      coin.x -= scroll
      if abs(coin.x - self.x) < 5 and abs(coin.y - self.y) < 5:
        reward += self.multiplier
        self.multiplier += 1
      elif coin.x < -3:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append(coin)
    self.coins = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(self.x // 10), 0] = True
    ordered = sorted(self.points, key=lambda point: point.x)
    heights = np.interp(np.arange(5, 100, 10), [point.x for point in ordered],
                        [point.y for point in ordered])
    for col, y in enumerate(heights):
      state[min(9, int(y // 10)), col, 1 if y < 50 else 2] = True
    state[5, :, 3] = True
    for coin in self.coins:
      if 0 <= coin.x < 100 and 0 <= coin.y < 100:
        state[int(coin.y // 10), int(coin.x // 10), 4] = True
    values = (self.steps, self.multiplier, self.held, round((self.vy + 5) * 40), round(self.vx * 100))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
