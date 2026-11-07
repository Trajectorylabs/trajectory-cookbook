from dataclasses import dataclass

import numpy as np


@dataclass
class Item:
  x: float
  y: float
  spike: bool = False


class GravelerEnvironment:
  score_cap = 300

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.walls = [(50.0, 60.0)] * 11
    self.center, self.width = 50.0, 60.0
    self.center_speed = self.width_speed = 0.0
    self.offset = 0.0
    self.y, self.vy = 65.0, 0.0
    self.items = [Item(15 + i * 7, 65) for i in range(5)]
    self.object_distance = 10.0
    self.coins_left, self.coin_y = 0, 65.0
    self.steps, self.multiplier = 0, 1
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
    center, width = self.walls[1]
    force = -0.015 if self.y < center else 0.015
    self.vy += force * (-1.5 if self.pressed else 1)
    self.y += self.vy
    self.vy *= 0.99
    self.terminal = abs(self.y - center) + 2 >= width / 2
    reward = 0
    remaining = []
    for item in self.items:
      if abs(item.x - 10) < 5 and abs(item.y - self.y) < 4:
        if item.spike:
          self.terminal = True
        else:
          reward += self.multiplier
          self.multiplier += 1
          continue
      if not item.spike and 0 <= item.x < 100:
        wall = int((item.x + self.offset) // 10)
        item_center, item_width = self.walls[wall]
        if abs(item.y - item_center) + 2 >= item_width / 2:
          self.coin_y += 10 if self.coin_y < 50 else -10
          continue
      item.x -= 0.5
      if item.x >= -3:
        remaining.append(item)
      elif not item.spike:
        self.multiplier = max(1, self.multiplier - 1)
    self.items = remaining
    self.offset += 0.5
    if self.offset >= 10:
      self.offset -= 10
      self._append_wall()
    self.object_distance -= 0.5
    if self.object_distance < 0:
      self._spawn_item()
    return reward

  def _append_wall(self) -> None:
    self.center_speed += self.random.uniform(-0.2, 0.2)
    self.width_speed += self.random.uniform(-0.2, 0.2)
    self.center += self.center_speed
    self.width += self.width_speed
    if self.center - self.width / 2 < 20 and self.center_speed < 0:
      self.center_speed += 1
    if self.center + self.width / 2 > 80 and self.center_speed > 0:
      self.center_speed -= 1
    if self.width < 32 and self.width_speed < 0:
      self.width_speed += 1
    if self.width > 60 and self.width_speed > 0:
      self.width_speed -= 1
    self.walls.pop(0)
    self.walls.append((self.center, self.width))

  def _spawn_item(self) -> None:
    if self.coins_left:
      self.items.append(Item(103, self.coin_y))
      self.coins_left -= 1
      self.object_distance = 7 if self.coins_left else self.random.uniform(40, 60)
    else:
      self.items.append(Item(103, self.center + self.random.uniform(-10, 10), spike=True))
      self.coins_left = int(self.random.integers(3, 7))
      self.object_distance = self.random.uniform(40, 60)
      side = self.random.choice((-1, 1))
      self.coin_y = self.center + side * self.width * self.random.uniform(0.2, 0.36)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), 1, 0] = True
    for col in range(10):
      center, width = self.walls[int((col * 10 + 5 + self.offset) // 10)]
      for row in range(10):
        state[row, col, 1] = abs(row * 10 + 5 - center) >= width / 2
      state[min(9, max(0, int(center) // 10)), col, 4] = True
    for item in self.items:
      if 0 <= item.x < 100 and 0 <= item.y < 100:
        state[int(item.y) // 10, int(item.x) // 10, 3 if item.spike else 2] = True
    velocity = min(511, max(0, round((self.vy + 2) * 100)))
    for row, value in enumerate((self.steps, self.multiplier, velocity)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
