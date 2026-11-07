from math import ceil, hypot

import numpy as np


class TappumpEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 10., 20.
    self.vy = self.radial_speed = self.coin_vy = 0.
    self.radius = 1.
    self.coin_y = 50.
    self.spike_distance, self.coin_distance = 40., 0.
    self.steps = 0
    self.held = self.terminal = False
    self.spikes, self.coins = [], []

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
    scroll = max(0, self.x - 20) * 0.2
    self.spike_distance -= scroll
    if self.spike_distance < 0:
      self.spikes.append((103, self.random.uniform(0, 99)))
      self.spike_distance += self.random.uniform(40, 140)
    self.spikes = [(x - scroll, y) for x, y in self.spikes if x - scroll >= -2]
    if tap:
      self.vy -= 2
    self.vy += -0.03 if self.held else 0.12
    if self.held:
      self.radial_speed += 0.08
      self.radius += self.radial_speed
    else:
      self.radius += (1 - self.radius) * 0.04
      self.radial_speed = 0.
    self.x += 1 - scroll
    self.y += self.vy
    self.terminal = not -5 - self.radius <= self.y <= 105 + self.radius
    self.terminal |= any(self._touches(x, y, 2.5) for x, y in self.spikes)
    self.coin_distance -= scroll
    self.coin_vy = (self.coin_vy + self.random.uniform(-0.1, 0.1)) * 0.98
    self.coin_y += self.coin_vy
    if (self.coin_y < 10 and self.coin_vy < 0) or (self.coin_y > 90 and self.coin_vy > 0):
      self.coin_vy *= -1
    if self.coin_distance < 0:
      self.coins.append((103, self.coin_y + self.random.uniform(-9, 9)))
      self.coin_distance += self.random.uniform(6, 9)
    reward = 0
    remaining = []
    for x, y in self.coins:
      x -= scroll
      if self._touches(x, y, 3):
        reward += ceil(self.radius)
      elif x >= -3 and not any(abs(x - sx) < 5.5 and abs(y - sy) < 5.5 for sx, sy in self.spikes):
        remaining.append((x, y))
    self.coins = remaining
    return reward

  def _touches(self, x, y, half_size):
    dx, dy = abs(x - self.x), abs(y - self.y)
    near = hypot(max(0, dx - half_size), max(0, dy - half_size))
    far = hypot(dx + half_size, dy + half_size)
    return near <= self.radius + 2.5 and far >= self.radius - 2.5

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(np.clip(self.y // 10, 0, 9))
    col = int(self.x // 10)
    state[row, col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 1] = abs(np.hypot(cols - self.x, rows - self.y) - self.radius) < 8
    for channel, objects in ((2, self.spikes), (3, self.coins)):
      for x, y in objects:
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    ahead = int(np.clip((self.y + 4 * self.vy) // 10, 0, 9))
    state[ahead, col, 4] = True
    values = (self.steps, self.held, round(self.radius * 4),
              round((self.vy + 20) * 20), round(self.radial_speed * 40))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(int(np.clip(value, 0, 1023)) & (1 << bit))
    return state
