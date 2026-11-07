from dataclasses import dataclass
from math import floor

import numpy as np


@dataclass
class Wall:
  y: int
  gap: float = 0.
  speed: float = 1.


class TwholsEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 50.
    self.vx = self.vy = 1
    self.walls = [Wall(9), Wall(79)]
    self.action = self.steps = 0
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
    turn = self.action == 5 and previous != 5
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(turn and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, turn):
    self.x += self.vx
    self.y += self.vy
    reward = 0
    for wall in self.walls:
      if wall.y - 2.5 < self.y < wall.y + 7.5 and (
        self.x - 2.5 < wall.gap or self.x + 2.5 > wall.gap + 40
      ):
        self.vy = 1 if self.y < 50 else -1
        reward = 1
      wall.gap -= wall.speed
      if wall.gap < -40:
        wall.gap = 99
        wall.speed = self.random.uniform(1, 2)
    if turn or self.x < 0 or self.x > 99:
      self.vx *= -1
    self.terminal = self.y < 0 or self.y > 99
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = (min(9, max(0, floor(value / 10))) for value in (self.y, self.x))
    state[row, col, 0] = True
    cols = np.arange(5, 100, 10)
    for wall in self.walls:
      wall_row = (wall.y + 2) // 10
      gap = (cols >= wall.gap) & (cols < wall.gap + 40)
      state[wall_row, :, 1] = ~gap
      state[wall_row, :, 2] = gap
    state[row, col, 3] = self.vx > 0
    state[row, col, 4] = self.vy > 0
    values = (self.steps, self.action, round((self.walls[0].gap + 40) * 3),
              round((self.walls[1].gap + 40) * 3),
              *(round(wall.speed * 100) for wall in self.walls))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
