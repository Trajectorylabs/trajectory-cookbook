from dataclasses import dataclass

import numpy as np


@dataclass
class Floor:
  x: float
  y: float
  kind: int


@dataclass
class Item:
  x: float
  y: float
  vy: float
  remaining: float
  travel: float
  gold: bool


class ScaffoldEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 5., 50.
    self.next_x, self.next_y = 57., 52.
    self.wall_x, self.wall_y = -9., 0.
    self.kind = self.steps = self.charge = 0
    self.direction = self.multiplier = 1
    self.item_distance = 0.
    self.held = self.terminal = False
    self.floors = [Floor(i * 6 + 3, 52, 1) for i in range(9)]
    self.items = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    if self.held and not held:
      if self.kind + self.direction not in (0, 1, 2):
        self.direction *= -1
      self.kind += self.direction
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
    scroll = 0.05 + max(0, self.next_x - 40) * 0.1
    self.wall_x -= scroll
    if self.wall_x < -6:
      self.wall_y = 47
      for _ in range(9):
        y = 47 + 4 * self.random.integers(-9, 9)
        if abs(y - self.next_y) < 25:
          self.wall_y = y
          break
      self.wall_x = 126 - self.next_x % 6
      self.items.append(Item(self.wall_x + 4, self.wall_y + 4, 0.1, 2, 4, True))
      self.item_distance += 30
    self.charge = self.charge + 1 if self.held else 0
    if self.charge == 16:
      self.floors.append(Floor(self.next_x, self.next_y, self.kind))
      self.next_x += 6
      self.next_y += 4 * (self.kind - 1)
      self.charge = 0
    for floor in self.floors:
      floor.x -= scroll
    self.floors = [floor for floor in self.floors if floor.x >= -3]
    self.next_x -= scroll
    self.x += max(0, 20 - self.x) * 0.2 - scroll
    self.terminal = self.y < 5 or self.y > 95 or (
      abs(self.x - (self.wall_x + 4)) < 4
      and (self.y - 3 < self.wall_y - 12 or self.y + 3 > self.wall_y + 18)
    )
    foot = self.x + 4
    surface = next((self._height(floor, foot) for floor in self.floors
                    if abs(foot - floor.x) <= 3), None)
    self.y += 9 if surface is None else max(-9, min(9, surface - self.y))
    self.item_distance -= scroll
    if self.item_distance < 0:
      gold = self.random.random() < 0.5
      travel = self.random.uniform(20, 60) / (4 if gold else 1.5)
      y = np.clip(self.next_y + self.random.uniform(-20, 20), 10, 90)
      self.items.append(Item(120, y, self.random.choice((-0.3, 0.3)), travel / 2, travel, gold))
      self.item_distance = self.random.uniform(15, 25)
    reward = 0
    remaining = []
    for item in self.items:
      item.x -= scroll
      item.y += item.vy
      item.remaining -= abs(item.vy)
      if item.remaining < 0:
        item.remaining = item.travel
        item.vy *= -1
      if abs(item.x - self.x) < 5 and abs(item.y - self.y) < 5:
        if item.gold:
          reward += self.multiplier
          self.multiplier += 1
          continue
        self.terminal = True
      if item.x >= -3:
        remaining.append(item)
      elif item.gold:
        self.multiplier = max(1, self.multiplier - 1)
    self.items = remaining
    return reward

  @staticmethod
  def _height(floor, x):
    return floor.y - 2 + (floor.kind - 1) * (x - floor.x + 3) * 2 / 3

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    entities = [(self.x, self.y, 0), (self.next_x, self.next_y, 2)]
    entities.extend((floor.x, floor.y, 1) for floor in self.floors)
    entities.extend((item.x, item.y, 3 if item.gold else 4) for item in self.items)
    for x, y, channel in entities:
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), channel] = True
    wall = self.wall_x + 4
    if 0 <= wall < 100:
      rows = np.arange(5, 100, 10)
      state[(rows < self.wall_y - 12) | (rows > self.wall_y + 18), int(wall // 10), 4] = True
    values = (self.steps, self.kind, self.direction > 0, self.charge, self.multiplier, self.held)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
