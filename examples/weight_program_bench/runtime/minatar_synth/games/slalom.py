from dataclasses import dataclass
from math import cos, floor, hypot, pi, sin, sqrt

import numpy as np


@dataclass
class Wall:
  y: float
  side: int
  length: int

  @property
  def tip(self):
    return 99 - self.length if self.side > 0 else self.length


@dataclass
class Rock:
  x: float
  y: float
  width: int
  height: int


class SlalomEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 90.0
    self.angle = -3 * pi / 4
    self.vx, self.vy = cos(self.angle), sin(self.angle)
    self.scroll = self.wall_distance = self.rock_distance = 0.
    self.side = 1
    self.closest = 99.
    self.steps = 0
    self.held = self.terminal = False
    self.walls, self.rocks = [], []
    self.target = None

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.rock_distance -= self.scroll
    if self.rock_distance < 0:
      x = self.random.integers(-5, 5) + 100 * self.random.integers(2)
      self.rocks.append(Rock(x, -9, self.random.integers(9, 19), 2 * self.random.integers(5, 9)))
      self.rock_distance += 10
    for rock in self.rocks:
      rock.y += self.scroll
    self.rocks = [rock for rock in self.rocks if rock.y < 109]
    self.wall_distance -= self.scroll
    if self.wall_distance < 0:
      self.walls.append(Wall(-5, self.side, self.random.integers(40, 60)))
      self.side *= -1
      self.wall_distance += self.random.integers(80, 90)
    for wall in self.walls:
      wall.y += self.scroll
    target = next((wall for wall in self.walls if wall.y < self.y + 9), None)
    self.walls = [wall for wall in self.walls if wall.y < 105]
    reward = 0
    if target is not self.target:
      reward = max(0, floor(100 / (sqrt(self.closest) + 1)) - 15)
      self.target = target
      self.closest = 99
    if self.held and target is not None:
      self.angle += target.side * 0.07
    self.vx = self.vx * 0.98 + 0.03 * cos(self.angle)
    self.vy = self.vy * 0.98 + 0.03 * sin(self.angle)
    self.x += self.vx
    self.y += self.vy
    self.scroll = max(0, 88 - self.y) * 0.5
    self.y += self.scroll
    self.terminal = not (0 <= self.x <= 99 and 0 <= self.y <= 99)
    self.terminal |= any(
      abs(wall.y - self.y) < 4 and (self.x > wall.tip - 2 if wall.side > 0 else self.x < wall.tip + 2)
      for wall in self.walls
    )
    self.terminal |= any(abs(rock.x - self.x) < rock.width / 2 + 2
                         and abs(rock.y - self.y) < rock.height / 2 + 2 for rock in self.rocks)
    if target is not None:
      self.closest = min(self.closest, hypot(target.tip - self.x, target.y - self.y))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = np.clip(np.array([self.y, self.x]) // 10, 0, 9).astype(int)
    state[row, col, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for wall in self.walls:
      state[:, :, 1] |= (abs(rows - wall.y) < 7) & (
        cols >= wall.tip - 5 if wall.side > 0 else cols <= wall.tip + 5)
    for rock in self.rocks:
      state[:, :, 2] |= (abs(rows - rock.y) < rock.height / 2 + 5) & (
        abs(cols - rock.x) < rock.width / 2 + 5)
    if self.target is not None and 0 <= self.target.y < 100:
      state[int(self.target.y // 10), self.target.tip // 10, 3] = True
    x, y = self.x + 10 * cos(self.angle), self.y + 10 * sin(self.angle)
    row, col = np.clip(np.array([y, x]) // 10, 0, 9).astype(int)
    state[row, col, 4] = True
    for row, value in enumerate((self.steps, self.held, round(self.closest * 5),
                                 round((self.vx + 2) * 100), round((self.vy + 2) * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
