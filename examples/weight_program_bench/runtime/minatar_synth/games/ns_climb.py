from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Wall:
  x: float
  count: int
  vx: float = 0
  pole: int = 0
  previous: int = -1


@dataclass
class Magnet:
  x: float
  y: float
  side: int
  pole: int


class NsClimbEnvironment:
  score_cap = 1500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.vx, self.vy = 50.0, 90.0, 0.0, -1.0
    self.pole = 1
    self.steps = 0
    self.pressed = self.terminal = False
    self.walls = [Wall(x, int(self.random.integers(2, 4))) for x in (20, 80)]
    self.magnets = [Magnet((20, 80)[i % 2], i * 5 - 5, i % 2, -1 if i < 10 else 0)
                    for i in range(22)]

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.pole *= -1
    self.pressed = pressed
    self.steps += 1
    reward = 0.0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self):
    self.x += self.vx
    self.y += self.vy
    self.vx *= 0.98
    self.vy += 0.001
    if self.y > 99:
      self.terminal = True
      return 0.0
    scroll = max(0, 70 - self.y) * 0.1
    self.y += scroll
    for magnet in self.magnets:
      magnet.y += scroll
      dx, dy = magnet.x - self.x, magnet.y - self.y
      distance = math.hypot(dx, dy)
      if magnet.pole and distance:
        force = -magnet.pole * self.pole / distance ** 3
        self.vx += dx * force
        self.vy += dy * force
      if abs(dx) < 8 and abs(dy) < 8:
        inward = 1 if magnet.side == 0 else -1
        self.x = magnet.x + 10 * inward
        if self.vx * inward < 0:
          self.vx *= -0.7
      if magnet.y > 105:
        self._recycle(magnet)
    return scroll

  def _recycle(self, magnet):
    magnet.y -= 110
    wall = self.walls[magnet.side]
    wall.x += wall.vx
    magnet.x = wall.x
    wall.vx += self.random.uniform(-0.1, 0.1)
    low = 10 + magnet.side * 50
    if (wall.x < low and wall.vx < 0) or (wall.x > low + 30 and wall.vx > 0):
      wall.vx *= -0.5
    wall.count -= 1
    if wall.count < 0:
      if wall.pole:
        wall.pole = 0
        wall.count = int(self.random.integers(2, 4))
      else:
        wall.pole = wall.previous = -wall.previous
        wall.count = int(self.random.integers(3, 9))
    magnet.pole = wall.pole

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row = min(9, max(0, int(self.y // 10)))
    col = min(9, max(0, int(self.x // 10)))
    state[row, col, 0] = True
    for magnet in self.magnets:
      if 0 <= magnet.x < 100 and 0 <= magnet.y < 100:
        state[int(magnet.y // 10), int(magnet.x // 10), 2 + magnet.pole] = True
    row = min(9, max(0, int((self.y + 4 * self.vy) // 10)))
    col = min(9, max(0, int((self.x + 4 * self.vx) // 10)))
    state[row, col, 4] = True
    for row, value in enumerate((self.steps, round((self.vy + 5) * 40), round((self.vx + 5) * 40))):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, :2, 5] = self.pressed, self.pole == 1
    return state
