from dataclasses import dataclass
from math import floor

import numpy as np


@dataclass
class Human:
  speed: float
  phase: int
  x: float = -4
  y: float = 89
  running: bool = False
  falling: bool = False


@dataclass
class Drop:
  x: float
  y: float
  vx: float
  strong: bool


class ShinyEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.ground = np.full(22, 8.0)
    self.cloud_x = np.arange(1, 31) * 199 / 31 + self.random.uniform(-3, 3, 30)
    self.cloud_y = self.random.uniform(5, 15, 30)
    self.cloud_radius = self.random.uniform(5, 10, 30)
    self.weak_timer = self.random.uniform(0, 99, 30)
    self.strong_timer = self.random.uniform(100, 900, 30)
    self.rain = self.steps = self.population = 0
    self.wind = 0.
    self.held = self.terminal = False
    self.humans, self.drops = [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
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

  def _cloud_positions(self):
    clear = np.where(self.cloud_x < 99, -20, 220)
    return clear + (self.cloud_x - clear) * self.rain / 10

  def _advance(self, tap):
    if not self.humans:
      self.population = min(9, self.population + 1)
      self.humans = [Human(self.random.uniform(0.1, 0.15), self.random.integers(999))
                     for _ in range(self.population)]
      self.ground[20] = 8
    if tap:
      self.wind = self.random.uniform(-0.5, 0.5)
      self.ground[20] -= 0.1
    self.rain = min(10, max(0, self.rain + (1 if self.held else -1)))
    cloud_positions = self._cloud_positions()
    for timer, strong, bounds in ((self.weak_timer, False, (100, 200)),
                                  (self.strong_timer, True, (500, 999))):
      timer -= 1
      for index in np.flatnonzero(timer < 0):
        if self.held:
          self.drops.append(Drop(cloud_positions[index], self.cloud_y[index] + self.cloud_radius[index],
                                 self.wind + self.random.uniform(-0.2, 0.2), strong))
        timer[index] = self.random.uniform(*bounds)
    support = self.ground >= 8
    support[20] = self.ground[20] >= 1
    self.ground[:20] = np.minimum(self.ground[:20] + 0.12, 8)
    self.ground[21] = min(8, self.ground[21] + 0.12)
    if support[20]:
      self.ground[20] -= 0.005
    count = len(self.humans)
    reward = 0
    survivors = []
    for human in self.humans:
      if not human.falling:
        human.x += human.speed * (5 if human.running else 1)
      on_ground = human.y < 102 and any(
        intact and abs(human.x - (index * 10 - 5)) < 7
        for index, intact in enumerate(support)
      )
      human.falling = not on_ground
      human.y = 89 if on_ground else human.y + 1
      if human.y > 103:
        self.population -= 1
        self.terminal |= self.population <= 0
        continue
      human.phase += 1
      if human.phase % 30 == 0:
        human.running = self.held
      if human.x > 203:
        reward += max(0, floor(self.ground[20] * count))
      else:
        survivors.append(human)
    self.humans = survivors
    drops = []
    for drop in self.drops:
      drop.x += drop.vx
      drop.y += 1
      if drop.strong and any(not h.falling and abs(h.x - drop.x) < 4
                             and abs(h.y - drop.y - 2) < 5 for h in self.humans):
        continue
      if drop.y <= 90:
        drops.append(drop)
      elif drop.strong:
        index = floor(drop.x / 10) + 1
        if 2 <= index < 20:
          self.ground[index] = 0
    self.drops = drops[-128:]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for human in self.humans:
      col = int(np.clip(human.x // 20, 0, 9))
      state[min(9, int(human.y // 10)), col, int(human.falling)] = True
    for index in range(1, 21):
      size = self.ground[index]
      if (index == 20 and size < 1) or (index != 20 and size < 8):
        state[9, (index - 1) // 2, 2] = True
    for drop in self.drops:
      if drop.strong and 0 <= drop.x < 200:
        state[int(drop.y // 10), int(drop.x // 20), 3] = True
    for x, y in zip(self._cloud_positions(), self.cloud_y):
      if 0 <= x < 200:
        state[int(y // 10), int(x // 20), 4] = True
    values = (self.steps, self.population, self.rain, self.held, round(max(0, self.ground[20]) * 50))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
