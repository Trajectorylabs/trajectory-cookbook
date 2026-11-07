from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Drop:
  x: float
  speed: float
  bonus: bool
  y: float = 2


class SpearainEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 50.
    self.direction = 1
    self.speed = self.spawn_rate = 1.
    self.timer = self.steps = self.bonus_count = 0
    self.held = self.terminal = False
    self.drops = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    press, release = held and not self.held, self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(press and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, press, release):
    self.timer -= 1
    if self.timer < 0:
      self.bonus_count -= 1
      bonus = self.bonus_count < 0
      self.drops.append(Drop(self.random.uniform(12, 88),
                             self.random.uniform(1, 2) / (3 if bonus else 1), bonus))
      if bonus:
        self.bonus_count = 12
      self.timer += self.random.uniform(40, 60) / self.spawn_rate
    if press:
      self.speed *= 0.5
    if self.speed < 1 and self.held:
      self.speed *= 0.98
      self.spawn_rate *= 1.05
    else:
      self.speed += (1 - self.speed) * 0.05
      self.spawn_rate = 1
    if self.speed < 0.03 or release:
      self.speed = 1 / sqrt(self.speed)
    # Collision uses the pre-movement position, where the source draws the drop.
    visible = [(drop.x, drop.y, drop.bonus) for drop in self.drops]
    for drop in self.drops:
      velocity = drop.speed * self.speed
      if drop.bonus and velocity > 78:
        velocity *= 0.2
      elif not drop.bonus and velocity > 74:
        velocity *= 1.5
      drop.y += velocity
    self.drops = [drop for drop in self.drops if drop.y < 87]
    self.x += self.direction / sqrt(self.speed)
    reward = 0
    if (self.x < 12 and self.direction < 0) or (self.x > 88 and self.direction > 0):
      self.direction *= -1
      reward = 1
    hits = [bonus for x, y, bonus in visible if abs(x - self.x) < 4 and abs(y - 87) < 5]
    if any(hits):
      reward += len(self.drops)
      self.drops.clear()
      self.spawn_rate = 1
      self.timer = 60
    elif hits:
      self.terminal = True
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, int(np.clip(self.x // 10, 0, 9)), 0] = True
    for drop in self.drops:
      state[int(drop.y // 10), int(drop.x // 10), 2 if drop.bonus else 1] = True
    state[9, :, 3] = True
    state[:, [0, 9], 3] = True
    x = self.x + 10 * self.direction
    state[8, int(np.clip(x // 10, 0, 9)), 4] = True
    values = (self.steps, self.held, round(self.speed * 100),
              self.bonus_count, round(max(0, self.timer)))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
