from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


LANES = (25, 50, 75)


@dataclass
class Falling:
  lane: int
  y: float = -5
  phase: float = 0


class StormveilEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 50.
    self.lane = 1
    self.squash = 1.
    self.steps = self.frame = self.timer = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.obstacles, self.coins = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if held and not self.held:
      self.lane = (self.lane + 1) % 3
      self.squash = 0.6
    self.held = held
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    vx = (LANES[self.lane] - self.x) * 0.5
    self.x += vx
    self.squash += (1 - self.squash) * 0.15
    breathe = 1 + sin(self.frame * 0.1) * 0.05
    tilt = vx * 0.08
    width, height = 3 * self.squash * breathe, 3 * (2 - self.squash) * breathe
    half_x = abs(cos(tilt)) * height + abs(sin(tilt)) * width
    half_y = abs(sin(tilt)) * height + abs(cos(tilt)) * width
    self.timer -= 1
    if self.timer < 0:
      blocked = int(self.random.integers(3))
      self.obstacles.append(Falling(blocked, phase=self.random.uniform(0, 2 * pi)))
      coin_lane = int(self.random.choice([lane for lane in range(3) if lane != blocked]))
      self.coins.append(Falling(coin_lane))
      self.timer += 40
    for obstacle in self.obstacles:
      obstacle.y += 1.5
      obstacle.phase += 0.2
      pulse = 1 + sin(obstacle.phase) * 0.15
      self.terminal |= self._touches(obstacle, half_x + 3 * pulse, half_y + 3 * (2 - pulse))
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y <= 105]
    reward = 0
    remaining = []
    for coin in self.coins:
      coin.y += 1.5
      coin.phase += 0.12
      if self._touches(coin, half_x + 1 + 2 * abs(cos(coin.phase)), half_y + 3):
        reward += self.multiplier
        self.multiplier = min(16, self.multiplier + 1)
      elif coin.y > 105:
        self.multiplier = 1
      else:
        remaining.append(coin)
    self.coins = remaining
    self.frame += 1
    return reward

  def _touches(self, item, half_x, half_y):
    return abs(LANES[item.lane] - self.x) < half_x and abs(item.y - 85) < half_y

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, int(self.x // 10), 0] = True
    for channel, items in ((1, self.obstacles), (2, self.coins)):
      for item in items:
        if 0 <= item.y < 100:
          state[int(item.y // 10), LANES[item.lane] // 10, channel] = True
    state[:, [x // 10 for x in LANES], 3] = True
    state[8, LANES[self.lane] // 10, 4] = True
    values = (self.steps, self.multiplier, self.held, self.timer, round(self.squash * 100))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
