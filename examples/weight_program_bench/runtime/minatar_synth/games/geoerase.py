from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Enemy:
  x: float
  y: float
  vx: float
  vy: float
  square: bool
  marked: bool = False


class GeoeraseEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.spawn_timer = 0
    self.angle = 0.0
    self.combo = 1
    self.pressed = self.terminal = False
    self.enemies: list[Enemy] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    release = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, release: bool) -> int:
    self.spawn_timer -= 1
    if self.spawn_timer <= 0:
      self._spawn()
      self.spawn_timer = 60
    self.angle = (self.angle + (0.04 if self.pressed else 0.08)) % math.tau
    reward = 0
    if release:
      marked = [enemy for enemy in self.enemies if enemy.marked]
      reward = self.combo * sum(2 if enemy.square else 1 for enemy in marked)
      self.combo = min(16, self.combo + len(marked) - 1) if len(marked) > 1 else max(1, self.combo - 1)
      self.enemies = [enemy for enemy in self.enemies if not enemy.marked]
    for enemy in self.enemies:
      if self.pressed:
        dx, dy = enemy.x - 50, enemy.y - 50
        difference = (math.atan2(dy, dx) - self.angle + math.pi) % math.tau - math.pi
        enemy.marked |= 3 < math.hypot(dx, dy) < 60 and abs(difference) < 0.2
      enemy.x += enemy.vx
      enemy.y += enemy.vy
      self.terminal |= abs(enemy.x - 50) < 5.5 and abs(enemy.y - 50) < 5.5
    return reward

  def _spawn(self) -> None:
    offset = self.random.uniform(0, 100)
    side = int(self.random.integers(4))
    x, y = ((offset, -5), (offset, 105), (-5, offset), (105, offset))[side]
    square = self.random.random() < 0.3
    speed = 0.4 if square else 0.25
    angle = math.atan2(50 - y, 50 - x)
    self.enemies.append(Enemy(x, y, speed * math.cos(angle), speed * math.sin(angle), square))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 100 and 0 <= enemy.y < 100:
        row, col = int(enemy.y) // 10, int(enemy.x) // 10
        state[row, col, 2 if enemy.square else 1] = True
        state[row, col, 3] |= enemy.marked
    for distance in range(0, 61, 4):
      x = 50 + math.cos(self.angle) * distance
      y = 50 + math.sin(self.angle) * distance
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, 4] = True
    for row, value in enumerate((self.steps, self.combo, round(self.angle * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
