from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Whale:
  x: float
  y: float
  lost: bool = False


@dataclass
class Ice:
  x: float
  y: float
  width: float
  height: float
  life: float
  age: int = 0

  def geometry(self):
    scale = 1 - (1 - min(1, self.age / 150)) ** 3
    if self.age > self.life - 120:
      scale *= max(0, 1 - ((self.age - self.life + 120) / 120) ** 2)
    height = self.height * scale
    return self.y + self.height - height, height


class PodMigrationEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lead = Whale(20, 50)
    self.pod = [Whale(10, 45), Whale(10, 55), Whale(5, 50)]
    self.ice: list[Ice] = []
    self.direction = self.steps = self.respawn = self.knockback = 0
    self.kx = self.ky = 0.0
    self.multiplier = 1.0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction = (self.direction + 1) % 4
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.multiplier = max(1, self.multiplier * 0.998)
    self.respawn += 1
    if self.respawn >= 300:
      whale = next((w for w in self.pod if w.lost), None)
      if whale is not None:
        whale.lost = False
        self.respawn = 0
    previous = self.lead
    for whale in self.pod:
      whale.x += (previous.x - whale.x) * 0.05
      whale.y += (previous.y - whale.y) * 0.05
      previous = whale
      if not whale.lost and abs(whale.x - self.lead.x) < 5 and abs(whale.y - self.lead.y) < 3:
        whale.lost = True
        self.respawn = 0
    chance = 0.03 if len(self.ice) < 3 else 0.02 if len(self.ice) < 5 else 0.01
    if self.random.random() < chance:
      x, y = self.random.uniform(10, 90, 2)
      if all(math.hypot(w.x - x, w.y - y) >= 20 for w in self.pod):
        self.ice.append(Ice(x, y, self.random.uniform(9, 16),
                            self.random.uniform(6, 10), self.random.uniform(500, 600)))
    blocked = False
    reward = 0
    remaining = []
    for ice in self.ice:
      if ice.age < ice.life:
        y, height = ice.geometry()
        if height and any(not w.lost and self._hits(w, ice, y, height) for w in self.pod):
          reward += math.floor(self.multiplier + 0.5)
          self.multiplier += 1
          continue
        if 150 <= ice.age < ice.life - 120:
          blocked |= self._hits(self.lead, ice, y, height)
        ice.age += 1
        remaining.append(ice)
    self.ice = remaining
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[self.direction]
    if self.knockback:
      self.lead.x += self.kx
      self.lead.y += self.ky
      self.kx *= 0.88
      self.ky *= 0.88
      self.knockback -= 1
    if blocked:
      self.kx, self.ky, self.knockback = -1.8 * dx, -1.8 * dy, 25
    else:
      self.lead.x += dx
      self.lead.y += dy
    self.lead.x = min(97, max(3, self.lead.x))
    self.lead.y = min(97, max(3, self.lead.y))
    self.terminal = all(whale.lost for whale in self.pod)
    return reward

  @staticmethod
  def _hits(whale, ice, y, height):
    return abs(whale.x - ice.x) < ice.width / 2 + 3 and abs(whale.y - y) < height / 2 + 2

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.lead.y // 10), int(self.lead.x // 10), 0] = True
    for whale in self.pod:
      state[int(whale.y // 10), int(whale.x // 10), 2 if whale.lost else 1] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for ice in self.ice:
      y, height = ice.geometry()
      if height:
        channel = 3 if 150 <= ice.age < ice.life - 120 else 4
        state[:, :, channel] |= (abs(cols - ice.x) < ice.width / 2 + 5) & (abs(rows - y) < height / 2 + 5)
    gauges = (self.steps, self.direction, min(511, self.respawn), self.knockback,
              min(511, round(self.multiplier * 10)))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
