from dataclasses import dataclass

import numpy as np


@dataclass
class Cave:
  width: float
  x: float = 0.
  vx: float = 0.
  vw: float = 0.


@dataclass
class Wall:
  left: float
  right: float
  y: float
  direction: int


@dataclass
class Gold:
  x: float
  y: float
  direction: int


class UdCaveEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = 0.
    self.wall_distance, self.gold_distance = -50., 5.
    self.caves = [Cave(20), Cave(15), Cave(15)]
    self.walls, self.gold = [], []
    self.multiplier = 1
    self.steps = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
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

  def _advance(self):
    self.wall_distance -= 1
    reward = 0
    if self.wall_distance < 0:
      reward = self.multiplier
      self._spawn()
      self.wall_distance += 10
    for wall in self.walls:
      wall.y += wall.direction
    self.walls = [wall for wall in self.walls if -10 <= wall.y <= 110]

    self.x = min(25, max(-25, self.x + (0.5 if self.action == 5 else -0.5)))
    players = ((25 + self.x, 90), (75 - self.x, 10))
    if any(self._hits_wall(x, y, 1.5, 2) for x, y in players):
      self.terminal = True
      return reward

    remaining = []
    for gold in self.gold:
      gold.y += gold.direction
      if self._hits_wall(gold.x, gold.y, 2, 2):
        continue
      if any(abs(x - gold.x) <= 4 and abs(y - gold.y) <= 4 for x, y in players):
        self.multiplier += 1
      elif -3 <= gold.y <= 103:
        remaining.append(gold)
    self.gold = remaining
    return reward

  def _spawn(self):
    center = self.caves[0]
    for index, cave in enumerate(self.caves):
      cave.vx += self.random.uniform(-2, 2)
      cave.vw += self.random.uniform(-1, 1)
      cave.x += cave.vx
      cave.width += cave.vw
      low, high = (-10, 10) if index == 0 else (center.x - center.width, center.x + center.width)
      if (cave.x - cave.width < low and cave.vx < 0) or (
        cave.x + cave.width > high and cave.vx > 0
      ):
        cave.vx *= -0.5
        cave.x += cave.vx
      minimum, maximum = (10, 14) if index == 0 else (center.width, 18)
      if (cave.width < minimum and cave.vw < 0) or (cave.width > maximum and cave.vw > 0):
        cave.vw *= -0.5
        cave.width += cave.vw

    for cave, mid, direction in zip(self.caves[1:], (25, 75), (1, -1), strict=True):
      y = -self.wall_distance if direction == 1 else 100 + self.wall_distance
      x = mid + direction * cave.x
      if x - cave.width > mid - 25:
        self.walls.append(Wall(mid - 25, x - cave.width, y, direction))
      if x + cave.width < mid + 25:
        self.walls.append(Wall(x + cave.width, mid + 25, y, direction))

    self.gold_distance -= 1
    if self.gold_distance < 0:
      left = self.random.random() < 0.5
      cave = self.caves[1 if left else 2]
      offset = self.random.uniform(-0.8, 0.8) * cave.width
      if left:
        self.gold.append(Gold(25 + cave.x + offset, -self.wall_distance - 5, 1))
      else:
        self.gold.append(Gold(75 - cave.x + offset, 105 + self.wall_distance, -1))
      self.gold_distance = self.random.uniform(3, 9)

  def _hits_wall(self, x, y, half_width, half_height):
    return any(
      wall.left - half_width < x < wall.right + half_width
      and abs(y - (wall.y - 4.5 * wall.direction)) < 4.5 + half_height
      for wall in self.walls
    )

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, min(9, int((25 + self.x) // 10)), 0] = True
    state[1, min(9, int((75 - self.x) // 10)), 1] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for wall in self.walls:
      state[:, :, 2 if wall.direction == 1 else 3] |= (
        (cols + 5 > wall.left) & (cols - 5 < wall.right)
        & (abs(rows - (wall.y - 4.5 * wall.direction)) < 9.5)
      )
    for gold in self.gold:
      if 0 <= gold.x < 100 and 0 <= gold.y < 100:
        state[int(gold.y // 10), int(gold.x // 10), 4] = True
    values = (self.steps, self.action, round((self.x + 25) * 2), self.multiplier,
              round(self.wall_distance + 51), round(self.gold_distance * 16))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
