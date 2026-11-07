from dataclasses import dataclass

import numpy as np


@dataclass
class Walker:
  x: float
  y: float
  lane: int
  dx: int = 0
  cooldown: float = 0
  dot_distance: float = 0


class NotTurnEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.player = Walker(50, 99, 2, cooldown=5)
    self.speed = 1.0
    self.enemy_timer = 20.0
    self.bar_distance = 0.0
    self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.bars = []
    self.enemies = []
    self.dots = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _move(self, walker, speed, vertical, turn):
    if walker.dx:
      walker.x += walker.dx * speed
      target = 10 + walker.lane * 20
      if (walker.x - target) * walker.dx >= 0:
        walker.x, walker.dx = target, 0
      return
    walker.y += vertical * speed
    walker.cooldown -= speed
    if turn and walker.cooldown < 0:
      for direction in (1, -1):
        lane = walker.lane + direction
        if 0 <= lane < 5 and any(
            abs(x - walker.x - 6 * direction) < 9 and abs(y - walker.y - 3) < 2
            for x, y in self.bars):
          walker.dx, walker.lane, walker.cooldown = direction, lane, 5
          break

  def _advance(self):
    player = self.player
    scroll = max(0, player.y - 20) * 0.1
    self.bar_distance -= scroll
    if self.bar_distance < 0:
      a, b = self.random.integers(4, size=2)
      self.bars.append((20 + a * 20, 102))
      if abs(a - b) > 1:
        self.bars.append((20 + b * 20, 102))
      self.bar_distance += self.random.uniform(9, 12)
    self.bars = [(x, y - scroll) for x, y in self.bars if y - scroll >= -1]
    self._move(player, self.speed * 0.2, 1, self.pressed)
    self.speed += ((4 if self.pressed else 1) - self.speed) * 0.2
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      lane = int(self.random.integers(5))
      self.enemies.append(Walker(10 + lane * 20, 103, lane))
      self.enemy_timer = self.random.uniform(100, 150)
    remaining = []
    for enemy in self.enemies:
      self._move(enemy, 0.3, -1, True)
      self.terminal |= abs(enemy.x - player.x) < 5 and abs(enemy.y - player.y) < 5
      enemy.y -= scroll
      enemy.dot_distance -= 0.3
      if enemy.dot_distance < 0:
        enemy.dot_distance += 6
        self.dots.append((enemy.x, enemy.y))
      if enemy.y >= -3:
        remaining.append(enemy)
      else:
        self.multiplier = max(1, self.multiplier - 1)
    self.enemies = remaining
    reward = 0
    remaining = []
    for x, y in self.dots:
      if abs(x - player.x) < 4 and abs(y - player.y) < 4:
        reward += self.multiplier
        self.multiplier += 1
      elif y - scroll >= -3:
        remaining.append((x, y - scroll))
    self.dots = remaining
    player.y -= scroll
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.player.y // 10))), int(self.player.x // 10), 0] = True
    state[:, 1::2, 1] = True
    for channel, points in ((2, self.bars), (3, [(e.x, e.y) for e in self.enemies]), (4, self.dots)):
      for x, y in points:
        if 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    values = (self.steps, min(511, self.multiplier), round(self.speed * 100),
              self.player.lane, self.player.dx + 1)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
