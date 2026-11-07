from dataclasses import dataclass

import numpy as np


@dataclass
class Enemy:
  x: float
  y: float
  vy: float
  vx: float = 0.
  timer: float = 0.


@dataclass
class Bomb:
  x: float
  y: float
  count: float
  vy: float = 0.


@dataclass
class Blast:
  x: float
  y: float
  height: float = 6.


class VBombEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.vx, self.target = 40., 1., 1
    self.enemies, self.bombs, self.blasts = [], [], []
    self.timer = 0.
    self.multiplier = 1
    self.steps = self.action = 0
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
    self.timer -= 1
    if self.timer < 0:
      x = -3 if self.random.random() < 0.5 else 103
      self.enemies.append(Enemy(x, self.random.uniform(70, 80), -self.random.uniform(0.01, 0.03)))
      self.timer = self.random.uniform(120, 150)
    if not self.blasts:
      self.multiplier = 1
    visible = [Blast(blast.x, blast.y, blast.height) for blast in self.blasts]
    for blast in self.blasts:
      blast.y -= 7
      blast.height += 5
    self.blasts = [blast for blast in self.blasts if blast.y + blast.height >= 0]
    reward = 0
    enemies = []
    for enemy in self.enemies:
      enemy.timer -= 1
      if enemy.timer < 0:
        if 1 < enemy.x < 99:
          self.bombs.append(Bomb(enemy.x, enemy.y, self.random.integers(3, 9) + 0.9))
        enemy.timer = self.random.uniform(200, 300)
      enemy.vx = min(0.1, max(-0.1, enemy.vx + (self.x - enemy.x) * 0.0002))
      if enemy.y < 10:
        enemy.vy = 0
      enemy.x += enemy.vx
      enemy.y += enemy.vy
      if any(_overlap(blast, enemy.x, enemy.y, 3, 3) for blast in visible):
        reward += self.multiplier
        self.multiplier += 1
      else:
        enemies.append(enemy)
    self.enemies = enemies

    bombs = []
    for bomb in self.bombs:
      bomb.vy += 0.1
      bomb.y = min(87, bomb.y + bomb.vy)
      fuse_y = bomb.y - (10 - bomb.count) * 2 - max(0, 1 - bomb.count) * 50
      if any(_overlap(blast, bomb.x, fuse_y, 2, 3) for blast in visible):
        bomb.count = 0
      bomb.count -= 1 / 60
      if bomb.count <= 0:
        self.blasts.append(Blast(bomb.x, bomb.y))
      else:
        bombs.append(bomb)
    self.bombs = bombs

    if turn or (self.x > 99 and self.target > 0) or (self.x < 1 and self.target < 0):
      self.target *= -1
    self.vx += (self.target - self.vx) * 0.1
    self.x += self.vx
    self.terminal = any(_overlap(blast, self.x, 10, 3, 3) for blast in visible) or any(
      abs(enemy.x - self.x) < 6 and abs(enemy.y - 10) < 6 for enemy in self.enemies
    )
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[1, int(np.clip(self.x // 10, 0, 9)), 0] = True
    for objects, channel in ((self.enemies, 1), (self.bombs, 2)):
      for obj in objects:
        if 0 <= obj.x < 100 and 0 <= obj.y < 100:
          state[int(obj.y // 10), int(obj.x // 10), channel] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for blast in self.blasts:
      state[:, :, 3] |= (abs(cols - blast.x) < 8) & (abs(rows - blast.y) < blast.height / 2 + 5)
    state[9, :, 4] = True
    values = (self.steps, self.action, self.target > 0, round((self.vx + 1) * 100),
              self.multiplier, round(self.timer))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _overlap(blast, x, y, half_width, half_height):
  return abs(blast.x - x) < 3 + half_width and abs(blast.y - y) < blast.height / 2 + half_height
