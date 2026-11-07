from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Meteor:
  x: float
  y: float
  vx: float
  vy: float = 0
  gold: list[int] = field(default_factory=list)
  history: deque = field(default_factory=lambda: deque(maxlen=200))


class GTailEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.x, self.y = 50, 50.0
    self.spawn_timer = 0.0
    self.multiplier = 1
    self.terminal = False
    self.meteors: list[Meteor] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    direction = -1 if action == 1 else 1 if action == 4 else 0
    reward = 0
    for _ in range(4):
      reward += self._advance(direction)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, direction: int) -> int:
    self.x = min(99, max(0, self.x + direction * 2))
    self.y += (50 - self.y) * 0.001
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self.meteors.append(Meteor(self.random.uniform(10, 90), -5, self.random.uniform(-0.5, 0.5),
                                 gold=list(range(10, 10 + 5 * int(self.random.integers(3, 8)), 5))))
      self.spawn_timer = self.random.uniform(99, 120) * math.sqrt(len(self.meteors))
    reward = 0
    survivors = []
    for meteor in self.meteors:
      meteor.vx = min(2, max(-2, meteor.vx + (self.x - meteor.x) / 999))
      meteor.vy = min(2, max(-2, meteor.vy + (self.y - meteor.y) / 999))
      meteor.x += meteor.vx
      meteor.y += meteor.vy
      meteor.history.appendleft((meteor.x, meteor.y))
      remaining = []
      for age in meteor.gold:
        if age < len(meteor.history) and self._touches(*meteor.history[age], radius=5):
          reward += self.multiplier
        else:
          remaining.append(age)
      meteor.gold = remaining
      if not meteor.gold:
        reward += self.multiplier
        self.multiplier += 1
      elif self._touches(meteor.x, meteor.y, radius=5.5):
        self.y += 10
      elif -100 <= meteor.x < 200 and -100 <= meteor.y < 200:
        survivors.append(meteor)
    self.meteors = survivors
    self.terminal = self.y > 99
    return reward

  def _touches(self, x: float, y: float, radius: float) -> bool:
    return abs(self.x - x) < radius and abs(self.y - y) < radius

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y) // 10), self.x // 10, 0] = True
    for meteor in self.meteors:
      if 0 <= meteor.x < 100 and 0 <= meteor.y < 100:
        state[int(meteor.y) // 10, int(meteor.x) // 10, 1] = True
      else:
        row = min(9, max(0, int(meteor.y) // 10))
        col = min(9, max(0, int(meteor.x) // 10))
        state[row, col, 4] = True
      for index, (x, y) in enumerate(meteor.history):
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, 2 if index in meteor.gold else 3] = True
    for row, value in enumerate((self.steps, self.multiplier, round(self.spawn_timer), self.x)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
