from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Mine:
  x: float
  fuse: int = 0


@dataclass
class Enemy:
  x: float
  vx: float


class MFieldEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 80.0, 87.0
    self.vx = self.vy = self.mine_distance = self.enemy_distance = 0.0
    self.jumps = self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.mines = []
    self.enemies = []
    self.blasts = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _in_blast(self, x, radius):
    return any(abs(x - center) < width + radius for center, _, width in self.blasts)

  def _advance(self, tap):
    scroll = max(0, self.x - 99) * 0.1
    self.blasts = [(x - scroll, age + 1, max(0, math.sin((age + 1) * 0.1) * 25))
                   for x, age, _ in self.blasts if age < 31]
    self.enemy_distance -= scroll
    if self.enemy_distance < 0:
      if self.random.random() < 0.7:
        self.enemies.append(Enemy(203, -0.4 * self.random.uniform(1, 2)))
      else:
        self.enemies.append(Enemy(-3, 0.4 * self.random.uniform(1.5, 2)))
      self.enemy_distance += self.random.uniform(20, 30)
    reward = 0
    surviving = []
    for enemy in self.enemies:
      enemy.x += enemy.vx - scroll
      if not self.jumps and abs(enemy.x - self.x) > 20 and (enemy.x - self.x) * enemy.vx > 0:
        enemy.vx *= -1
      if self._in_blast(enemy.x, 3):
        reward += self.multiplier
        self.multiplier = min(64, self.multiplier * 2)
      elif enemy.x >= -9:
        surviving.append(enemy)
    self.enemies = surviving
    self.vx += ((1 if self.pressed else 0.3) - self.vx) * 0.1
    if self.jumps:
      self.vy += 0.05 if self.pressed else 0.1
      if self.y > 87:
        self.y, self.vy, self.jumps = 87, 0, 0
    if tap and self.jumps < 2:
      self.jumps += 1
      self.vy = -2
      self.y -= 6
    self.x += self.vx - scroll
    self.y += self.vy
    self.terminal = abs(self.y - 86) < 7 and self._in_blast(self.x, 3)
    if abs(self.y - 87) < 6 and any(abs(self.x - enemy.x) < 5 for enemy in self.enemies):
      if self.vy > 0:
        self.y = 80
        self.vy *= -0.8
      else:
        self.terminal = True
    self._move_mines(scroll)
    return reward

  def _move_mines(self, scroll):
    self.mine_distance -= scroll
    if self.mine_distance < 0:
      self.mines.append(Mine(203))
      self.mine_distance = (self.random.uniform(9, 20) if self.random.random() < 0.6
                            else self.random.uniform(50, 80))
    remaining = []
    chain = []
    for mine in self.mines:
      mine.x -= scroll
      if mine.fuse:
        mine.fuse += 1
      if mine.fuse > 59 or self._in_blast(mine.x, 3):
        chain.append((mine.x, 0, 0))
      elif mine.x >= -3:
        if not mine.fuse and abs(mine.x - self.x) < 6 and abs(91 - self.y) < 4.5:
          mine.fuse = 1
          self.multiplier = 1
        remaining.append(mine)
    self.mines = remaining
    self.blasts.extend(chain)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), int(self.x // 20), 0] = True
    for mine in self.mines:
      if 0 <= mine.x < 200:
        state[9, int(mine.x // 20), 2 if mine.fuse else 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200:
        state[8, int(enemy.x // 20), 3] = True
    for x, _, width in self.blasts:
      state[8:10, :, 4] |= abs(np.arange(10, 200, 20) - x) < width + 10
    for row, value in enumerate((self.steps, self.multiplier, self.jumps, round((self.vy + 4) * 40))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
