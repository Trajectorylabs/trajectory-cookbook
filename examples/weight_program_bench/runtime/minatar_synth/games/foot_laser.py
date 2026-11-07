from dataclasses import dataclass

import numpy as np


GROUND = 90
PLAYER_X = 20


@dataclass
class Enemy:
  x: float
  y: float
  vx: float
  flying: bool = False


class FootLaserEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.enemy_timer = 0
    self.wall_timer = self.random.uniform(300, 400)
    self.flying_timer = self.random.uniform(200, 300)
    self.y, self.vy = 50.0, 0.0
    self.jumps, self.multiplier = 9, 1
    self.pressed = self.grounded = self.terminal = False
    self.enemies: list[Enemy] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool) -> int:
    if not self.grounded:
      self.vy += 0.1 if self.pressed else 0.3
      self.y += self.vy
      if self.y > GROUND:
        self.y, self.vy = GROUND, 0
        self.grounded = True
        self.jumps, self.multiplier = 0, 1
    if tap:
      if self.jumps < 2:
        self.vy, self.grounded = -3, False
      elif self.jumps == 2:
        self.vy += 9
      self.jumps = min(9, self.jumps + 1)

    self._spawn()
    reward = 0
    survivors = []
    for enemy in self.enemies:
      enemy.x += enemy.vx
      distance = abs(enemy.x - PLAYER_X)
      if not self.grounded and distance < 4 and enemy.y > self.y + 2:
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= distance < 5 and abs(enemy.y - self.y) < 5
        if enemy.x >= -6:
          survivors.append(enemy)
    self.enemies = survivors
    return reward

  def _spawn(self) -> None:
    self.enemy_timer -= 1
    self.wall_timer -= 1
    self.flying_timer -= 1
    if self.enemy_timer < 0:
      self.enemies.append(Enemy(200, GROUND, -self.random.uniform(1, 2)))
      self.enemy_timer = self.random.uniform(30, 60)
    if self.wall_timer < 0:
      vx = -self.random.uniform(1, 2)
      for index in range(int(self.random.integers(3, 6))):
        self.enemies.append(Enemy(200, GROUND - 6 * index, vx))
      self.wall_timer = self.random.uniform(100, 600)
      self.enemy_timer += 9
    if self.flying_timer < 0:
      vx, y = -self.random.uniform(1, 2), self.random.uniform(50, 80)
      for index in range(int(self.random.integers(1, 5))):
        self.enemies.append(Enemy(206 + 7 * index, y, vx, flying=True))
      self.flying_timer = self.random.uniform(100, 400)
      self.enemy_timer += 9

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = max(0, int(self.y) // 10)
    state[row, 1, 0] = True
    if not self.grounded:
      state[row:, 1, 3] = True
    state[9, :, 4] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 200:
        state[int(enemy.y) // 10, int(enemy.x) // 20, 2 if enemy.flying else 1] = True
    for row, value in enumerate((self.steps, self.jumps, self.multiplier, round(self.vy * 10) + 128)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.grounded
    return state
