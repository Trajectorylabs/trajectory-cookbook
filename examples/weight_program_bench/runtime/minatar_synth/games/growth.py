from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Enemy:
  x: float
  size: float

  @property
  def screen_x(self) -> float:
    return self.x / (1 + max(0, self.x - 100) / 300)


class GrowthEnvironment:
  score_cap = 3500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.speed, self.size = 9.0, 1.0, 5.0
    self.spawn_distance = 0.0
    self.steps = 0
    self.pressed = self.terminal = False
    self.enemies: list[Enemy] = []

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
    scroll = max(0, (self.x - 9) * 0.5)
    target = 50 if self.pressed else 5
    self.size += (target - self.size) * min(999, max(1, self.speed)) * 0.01
    self.speed += (15 / self.size - 1) * 0.02
    self.x += self.speed - scroll
    front = self.x + self.size / 2
    if front < 1:
      self.terminal = True
      return 0
    self.spawn_distance -= scroll
    if self.spawn_distance < 0:
      size = 3.0
      if self.random.random() >= 0.8:
        width, height = self.random.uniform(0, 5, 2)
        size = width * height + 3
      self.enemies.append(Enemy(400, size if size >= 7 else 3))
      self.spawn_distance += self.random.uniform(30, 50)
    reward = 0
    remaining = []
    for enemy in self.enemies:
      enemy.x -= scroll
      if enemy.screen_x >= front:
        remaining.append(enemy)
      elif enemy.size > self.size:
        self.terminal = True
        break
      else:
        reward += math.ceil(min(999, max(1, self.speed)) * enemy.size)
    if not self.terminal:
      self.enemies = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    top = max(0, int((60 - self.size) // 7))
    front = min(9, max(0, int((self.x + self.size / 2) // 20)))
    state[top:9, :front + 1, 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.screen_x <= 200:
        col = min(9, int(enemy.screen_x // 20))
        state[int((60 - enemy.size) // 7):9, col, 1 if enemy.size <= self.size else 2] = True
    state[9, :, 3] = True
    if self.enemies:
      nearest = self.enemies[0]
      state[8, min(9, max(0, int(nearest.screen_x // 20))), 4] = True
    speed = min(511, max(0, round((self.speed + 5) * 8)))
    for row, value in enumerate((self.steps, round(self.size * 10), speed)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
