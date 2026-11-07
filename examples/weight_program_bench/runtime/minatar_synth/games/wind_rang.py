from dataclasses import dataclass

import numpy as np


@dataclass
class Flyer:
  x: float
  y: float
  vy: float
  vx: float = 0.


class WindRangEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.boomerangs, self.enemies = [], []
    self.throw_timer, self.enemy_timer = 18, 77.
    self.wind = self.multiplier = 1
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
    if self.action == 5 and previous != 5:
      self.wind *= -1
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.throw_timer -= 1
    if self.throw_timer <= 0:
      self.boomerangs.append(Flyer(50, 98, -1.5))
      self.throw_timer = 18
    reward = 0
    visible = []
    remaining = []
    for flyer in self.boomerangs:
      flyer.vx += self.wind * 0.08
      flyer.x += flyer.vx
      flyer.y += flyer.vy
      if flyer.y < 10:
        flyer.vy = 1.5
      visible.append(flyer)
      if flyer.vy > 0 and flyer.y > 93 and abs(flyer.x - 50) < 8:
        self.multiplier += 1
        reward += 10 * self.multiplier
      elif -5 <= flyer.x <= 105 and flyer.y <= 105:
        remaining.append(flyer)
    self.boomerangs = remaining

    self.enemy_timer -= 1
    if self.enemy_timer <= 0:
      self.enemies.append(Flyer(self.random.uniform(10, 90), 0, 0.25))
      self.enemy_timer = 77 * self.random.uniform(0.5, 2)
    remaining = []
    for enemy in self.enemies:
      enemy.vx += self.wind * 0.02
      enemy.x += enemy.vx
      if enemy.x < 0:
        enemy.x = 100
      elif enemy.x > 100:
        enemy.x = 0
      enemy.y += enemy.vy
      self.terminal |= enemy.y > 100
      if any(abs(enemy.x - b.x) < 5 and abs(enemy.y - b.y) < 5 for b in visible):
        reward += self.multiplier
      else:
        self.terminal |= enemy.y > 93 and abs(enemy.x - 50) < 5
        remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, 5, 0] = True
    for flyer in self.boomerangs:
      _mark(state, flyer, 1 if flyer.vy < 0 else 2)
    for enemy in self.enemies:
      _mark(state, enemy, 3)
    state[0, 9 if self.wind > 0 else 0, 4] = True
    values = (self.steps, self.action, self.multiplier, self.throw_timer, round(self.enemy_timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _mark(state, flyer, channel):
  if 0 <= flyer.x < 100 and 0 <= flyer.y < 100:
    state[int(flyer.y // 10), int(flyer.x // 10), channel] = True
