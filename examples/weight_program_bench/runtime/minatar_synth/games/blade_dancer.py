from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
PLAYER_X = 35
GROUND = 80
JUMP_DURATION = 30


@dataclass
class Enemy:
  x: float
  y: float
  size: float


class BladeDancerEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.pressed = self.terminal = False
    self.jump_ticks = 0.0
    self.y = float(GROUND)
    self.angle = math.pi / 2
    self.length = 10.0
    self.multiplier = 1
    self.enemies: list[Enemy] = []
    self.platforms = [(0.0, 200.0)]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed and not self.jump_ticks:
      self.jump_ticks = JUMP_DURATION
      self.multiplier = 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _sword_vector(self) -> tuple[float, float]:
    return self.length * math.cos(self.angle), self.length * math.sin(self.angle)

  def _advance_frame(self) -> int:
    if self.jump_ticks:
      self.jump_ticks = max(0, self.jump_ticks - (0.5 if self.pressed else 1))
      progress = self.jump_ticks / JUMP_DURATION
      self.y = GROUND - math.sin(progress * math.pi) * 30
      self.angle = self.angle - 0.3 if self.jump_ticks else math.pi / 2
      self.length = 8 + (1 - progress) * 22 if self.jump_ticks else 10

    if not self.enemies or self.random.random() < 0.04:
      if len(self.enemies) == 32:
        self.enemies.pop(0)
      self.enemies.append(Enemy(205, self.random.uniform(30, 75), self.random.uniform(3, 6)))
    reward = 0
    survivors = []
    sword_x, sword_y = self._sword_vector()
    sword_radius = 2.5 if self.jump_ticks else 1
    for enemy in self.enemies:
      enemy.x -= 1
      dx, dy = enemy.x - PLAYER_X, enemy.y - (self.y - 4)
      fraction = min(1, max(0, (dx * sword_x + dy * sword_y) / self.length**2))
      distance = math.hypot(dx - fraction * sword_x, dy - fraction * sword_y)
      radius = enemy.size / 2
      if distance <= radius + sword_radius:
        reward += self.multiplier
        self.multiplier += 1
        continue
      if abs(dx) < 2 + radius and abs(enemy.y - self.y) < 4 + radius:
        self.terminal = True
      if enemy.x >= -10:
        survivors.append(enemy)
    self.enemies = survivors

    self.platforms = [(left - 1, right - 1) for left, right in self.platforms if right >= 1]
    end = self.platforms[-1][1]
    if end < 200:
      gap = self.random.uniform(15, 30) if self.random.random() < 0.32 else 0
      length = self.random.uniform(30, 100) if gap else self.random.uniform(50, 150)
      self.platforms.append((end + gap, end + gap + length))
    if not self.jump_ticks and not any(left <= PLAYER_X <= right for left, right in self.platforms):
      self.terminal = True
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, column = int(self.y) // 10, PLAYER_X // 20
    state[row, column, 0] = True
    state[row, column, 4] = self.jump_ticks > 0
    sword_x, sword_y = self._sword_vector()
    for fraction in np.linspace(0, 1, 12):
      y = int(self.y - 4 + fraction * sword_y) // 10
      x = int(PLAYER_X + fraction * sword_x) // 20
      state[y, x, 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200:
        state[int(enemy.y) // 10, int(enemy.x) // 20, 2] = True
    for column in range(10):
      x = 20 * column + PLAYER_X % 20
      state[8, column, 3] = any(left <= x <= right for left, right in self.platforms)
    state[9, 0, 5] = self.pressed
    for row, value in enumerate((int(self.jump_ticks * 2), self.multiplier, self.steps)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
