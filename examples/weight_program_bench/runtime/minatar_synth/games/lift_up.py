from dataclasses import dataclass

import numpy as np


@dataclass
class Wall:
  x: float
  y: float
  width: float


@dataclass
class Item:
  x: float
  y: float
  bonus: bool


class LiftUpEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.walls = [Wall(50, i * 6 - 3, 60) for i in range(19)]
    self.top_wall = self.walls[0]
    self.wall_vx = self.wall_vw = 0.0
    self.x, self.y, self.target_y = 40.0, 90.0, 90.0
    self.direction = self.item_side = self.multiplier = 1
    self.ratio = self.item_distance = self.bonus_x = 0.0
    self.bonus_count = self.steps = 0
    self.items: list[Item] = []
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
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

  def _advance(self) -> int:
    self.ratio += (self.pressed - self.ratio) * 0.1
    scroll = (1 + self.ratio * 3) * 0.2
    for index, wall in enumerate(self.walls):
      wall.y += scroll
      if wall.y > 110:
        self._recycle_wall(index, wall)
    self.item_distance -= scroll
    if self.item_distance < 0:
      self._spawn_item()
    self.x += self.direction * 0.05 * (4 - self.ratio * 3)
    self.target_y -= 0.04
    self.y += (self.target_y - self.y) * 0.2
    self.terminal = self.y < 0 or self._hits_wall(self.x, self.y)
    reward = 0
    remaining = []
    for item in self.items:
      item.y += scroll
      if self._hits_wall(item.x, item.y):
        item.x += -1 if item.x > self.top_wall.x else 1
      if abs(item.x - self.x) < 5 and abs(item.y - self.y) < 5:
        if item.bonus:
          reward += self.multiplier
          self.multiplier = min(99, self.multiplier + 1)
          self.target_y += (99 - self.target_y) * 0.1
        else:
          self.direction *= -1
      elif item.y <= 103:
        remaining.append(item)
      elif item.bonus:
        self.multiplier = max(1, self.multiplier - 1)
    self.items = remaining
    return reward

  def _hits_wall(self, x: float, y: float) -> bool:
    return any(abs(y - wall.y) < 5 and abs(x - wall.x) + 3 > wall.width / 2
               for wall in self.walls)

  def _recycle_wall(self, index: int, wall: Wall) -> None:
    previous = self.walls[index - 1]
    wall.y -= 114
    wall.x, wall.width = previous.x + self.wall_vx, previous.width + self.wall_vw
    self.top_wall = wall
    self.wall_vx = min(5, max(-5, self.wall_vx + self.random.uniform(-1, 1))) * 0.95
    self.wall_vw = min(3, max(-3, self.wall_vw + self.random.uniform(-1, 1))) * 0.95
    if ((wall.x + wall.width / 2 > 95 and self.wall_vx > 0)
        or (wall.x - wall.width / 2 < 5 and self.wall_vx < 0)):
      self.wall_vx *= -0.5
    if (wall.width > 80 and self.wall_vw > 0) or (wall.width < 40 and self.wall_vw < 0):
      self.wall_vw *= -0.5

  def _spawn_item(self) -> None:
    wall = self.top_wall
    if self.bonus_count > 0:
      self.items.append(Item(self.bonus_x, -3, True))
      self.item_distance = 6
    elif self.bonus_count < 0 and self.random.random() < 0.5:
      self.bonus_count = int(self.random.integers(2, 6))
      self.bonus_x = wall.x + self.random.uniform(-0.25, 0.25) * wall.width
      self.item_distance = 0
    else:
      x = wall.x + self.random.uniform(0.1, 0.4) * self.item_side * wall.width
      self.items.append(Item(x, -3, False))
      if self.random.random() < 0.8:
        self.item_side *= -1
      self.item_distance = self.random.uniform(10, 20)
    self.bonus_count -= 1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = min(9, max(0, int(self.y // 10))), min(9, max(0, int(self.x // 10)))
    state[row, col, 0] = True
    for wall in self.walls:
      if 0 <= wall.y < 100:
        state[int(wall.y // 10), :, 1] |= abs(np.arange(5, 100, 10) - wall.x) > wall.width / 2
    for item in self.items:
      if 0 <= item.x < 100 and 0 <= item.y < 100:
        state[int(item.y // 10), int(item.x // 10), 2 + item.bonus] = True
    state[row, min(9, max(0, col + self.direction)), 4] = True
    for row, value in enumerate((self.steps, self.multiplier, round(self.ratio * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
