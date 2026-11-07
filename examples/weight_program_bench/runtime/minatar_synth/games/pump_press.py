from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class Enemy:
  x: float = 205
  size: float = 0
  attachment: float | None = None


class PumpPressEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.ship = 100.0
    self.speed = 1.0
    self.shot = self.rocket = None
    self.shot_range = self.steps = 0
    self.held = self.launched = self.terminal = False
    self.enemy_distance, self.rocket_distance = 0.0, 30.0
    self.enemies = []

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

  def _advance(self, tap):
    self.ship += self.speed
    self.speed += ((1 if self.held else 0.5) - self.speed) * 0.1
    scroll = 0.1 + max(0, self.ship - 100) * 0.1
    self.ship -= scroll
    if self.shot is None and tap:
      self.shot = self.ship + 6
      self.shot_range = 60
    shot = self.shot
    if shot is not None:
      shot += 3 - scroll
      self.shot_range -= 3
      self.shot = shot if self.shot_range >= 0 else None
    self.enemy_distance -= scroll
    if self.enemy_distance < 0:
      self.enemies.append(Enemy())
      self.enemy_distance = self.random.uniform(33, 43)
    reward = 0
    if self.rocket is None:
      self.rocket_distance -= scroll
      if self.rocket_distance < 0:
        self.rocket = 205
        self.launched = False
        self.rocket_distance = 100
    else:
      self.rocket -= scroll + int(self.launched)
      if not self.launched and shot is not None and abs(self.rocket - shot) < 5.5:
        self.launched = True
        self.shot = None
      if self.rocket < 5:
        count = sum(enemy.attachment is not None for enemy in self.enemies)
        reward = count * (count + 1) // 2
        self.rocket = None
        self.launched = False
    remaining = []
    for enemy in self.enemies:
      if enemy.attachment is not None:
        if self.rocket is not None:
          enemy.x = self.rocket + enemy.attachment
          remaining.append(enemy)
        continue
      if enemy.size > 0:
        enemy.size = max(0, enemy.size - 0.02)
      else:
        enemy.x += 0.5 if enemy.x < self.ship else -0.5
      enemy.x -= scroll
      radius = ceil(enemy.size) * 2 + 2
      if shot is not None and abs(enemy.x - shot) < radius + 3:
        self.shot = None
        enemy.size = ceil(enemy.size) + 1
        if enemy.size > 3:
          continue
      if self.launched and abs(enemy.x - self.rocket) < radius + 2.5:
        enemy.attachment = self.random.uniform(-6, -2)
      if enemy.size == 0 and abs(enemy.x - self.ship) < radius + 3:
        self.terminal = True
      if enemy.x >= -5:
        remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for x, row, channel in ((self.ship, 5, 0), (self.shot, 5, 1),
                            (self.rocket, 5, 2), (self.rocket if self.launched else None, 4, 2)):
      if x is not None and 0 <= x < 200:
        state[row, int(x // 20), channel] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200:
        row = 4 if enemy.attachment is not None else 5
        state[row, int(enemy.x // 20), 3 + int(enemy.size > 0)] = True
    for row, value in enumerate((self.steps, round(self.speed * 100),
                                 max(0, self.shot_range), self.held)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
