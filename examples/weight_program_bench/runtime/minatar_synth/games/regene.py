from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np


@dataclass
class Tile:
  x: float
  y: float
  seed: bool = False


@dataclass
class Ball:
  x: float
  y: float
  vx: float
  vy: float
  hits: int = 0


class RegeneEnvironment:
  score_cap = 1000
  moves = ((0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 50
    self.steps = self.action = 0
    self.distance = 0.0
    self.terminal = False
    self.tiles = []
    self.erased = []
    self.balls = [self._ball(20, 50)]

  def _ball(self, x, y):
    angle = self.random.uniform(-0.1, 0.1) * pi + self.random.choice((-1, 1)) * 0.7 * pi
    return Ball(x, y, cos(angle), sin(angle))

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self, population):
    width = self.random.integers(3, 7)
    gap = 0 if self.random.random() < 0.4 else self.random.integers(2, 6)
    top = int(gap + self.random.random() * (8 - 2 * gap))
    for col in range(width):
      for row in range(8):
        if not top <= row < top + gap:
          self.tiles.append(Tile(105 + col * 10 - self.distance, 15 + row * 10,
                                 self.random.random() < 0.1 / population))
    self.distance += (width + self.random.integers(3)) * 10

  def _advance(self):
    dx, dy = self.moves[self.action]
    self.x = min(99, max(0, self.x + 3 * dx))
    self.y = min(99, max(0, self.y + 3 * dy))
    population = sqrt(len(self.balls))
    scroll = 0.05 * (population + max(0, self.x - 90))
    self.distance -= scroll
    if self.distance < 0:
      self._spawn(population)
    for tile in self.erased:
      tile.x -= scroll
    self.erased = [tile for tile in self.erased if tile.x >= 14]
    remaining = []
    for tile in self.tiles:
      tile.x -= scroll
      reach = 5.5 if tile.seed else 6.5
      if abs(tile.x - self.x) < reach and abs(tile.y - self.y) < reach:
        if not tile.seed:
          self.erased.append(tile)
          continue
        if tile.x < 97:
          self.balls.append(self._ball(tile.x, tile.y))
          continue
      if tile.x >= (13 if tile.seed else 14):
        remaining.append(tile)
    overflow = max(0, len(self.erased) - 9)
    restored = self.erased[:overflow]
    solids = [(tile.x, tile.y) for tile in remaining if not tile.seed]
    self.tiles = remaining + restored
    self.erased = self.erased[overflow:]
    reward = 0
    survivors = []
    for ball in self.balls:
      ball.x -= scroll
      hit_x = self._blocked(ball.x + ball.vx, ball.y, solids)
      hit_y = self._blocked(ball.x, ball.y + ball.vy, solids)
      if hit_x:
        ball.vx *= -1
        ball.x += 2 * ball.vx
      if hit_y:
        ball.vy *= -1
        ball.y += 2 * ball.vy
      if ball.x < 14:
        ball.x = 14
        ball.vx = abs(ball.vx)
      if hit_x or hit_y:
        if not ball.hits:
          reward += len(self.balls)
        ball.hits += 1
      else:
        ball.hits = 0
      if ball.hits <= 9:
        ball.x += ball.vx
        ball.y += ball.vy
        survivors.append(ball)
    self.balls = survivors
    self.terminal = not survivors
    return reward

  @staticmethod
  def _blocked(x, y, solids):
    return x < 13 or x > 97 or y < 13 or y > 87 or any(
      abs(x - sx) < 7 and abs(y - sy) < 7 for sx, sy in solids
    )

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.y // 10, self.x // 10, 0] = True
    for tile in self.tiles:
      if 0 <= tile.x < 100:
        state[int(tile.y // 10), int(tile.x // 10), 1 + int(tile.seed)] = True
    for ball in self.balls:
      x = int(np.clip(ball.x // 10, 0, 9))
      y = int(np.clip(ball.y // 10, 0, 9))
      state[y, x, 3] = True
    if len(self.erased) == 9:
      oldest = self.erased[0]
      if 0 <= oldest.x < 100:
        state[int(oldest.y // 10), int(oldest.x // 10), 4] = True
    for row, value in enumerate((self.steps, len(self.erased), self.action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
