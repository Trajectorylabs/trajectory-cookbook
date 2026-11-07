from dataclasses import dataclass

import numpy as np


@dataclass
class Mover:
  x: float
  y: float
  vx: float
  vy: float

  def move(self) -> None:
    self.x += self.vx
    self.y += self.vy


class DPistolsEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.hold = self.spawn_timer = self.multiplier = 0
    self.pressed = self.terminal = False
    self.y, self.direction = 20.0, 1
    self.enemy_y, self.enemy_direction = 50.0, 1
    self.enemies: list[Mover] = []
    self.shots: list[Mover] = []

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
      reward += self._advance_frame(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, tap: bool) -> int:
    if tap:
      self.direction *= -1
      self.shots.extend(Mover(50, self.y + self.direction, vx, 0) for vx in (-2, 2))
      self.multiplier = max(0, self.multiplier - 1)
    self.hold = self.hold + 1 if self.pressed else 0
    if self.hold > 30:
      self.shots.extend(Mover(50, self.y, vx, vy)
                        for vx, vy in ((-2, 0), (2, 0), (0, -2), (0, 2)))
      self.hold = 0
      self.multiplier = max(1, self.multiplier - 5)
    self.y += self.direction * (1 - self.hold / 30)
    if not 0 <= self.y <= 99:
      self.direction *= -1
      self.y = min(99, max(0, self.y))
    for shot in self.shots:
      shot.move()
    self.shots = [shot for shot in self.shots if -3 <= shot.x < 103 and -3 <= shot.y < 103]

    self.spawn_timer = self.spawn_timer - 1 if self.enemies else -1
    if self.spawn_timer < 0:
      if self.random.random() < 0.3:
        self.enemy_y = self.random.uniform(9, 90)
        self.enemy_direction *= -1
      x = -3 if self.enemy_direction > 0 else 103
      self.enemies.append(Mover(x, self.enemy_y, self.enemy_direction * 0.3, 0))
      self.spawn_timer = int(self.random.integers(20, 41))

    survivors = []
    reward = 0
    for enemy in self.enemies:
      enemy.move()
      if enemy.vx and (enemy.x - 50) * enemy.vx >= 0:
        enemy.x = 50
        enemy.vy = -0.3 if self.y < enemy.y else 0.3
        enemy.vx = 0
      if any(
        abs(shot.x - enemy.x) < (3 if shot.vx else 6)
        and abs(shot.y - enemy.y) < (6 if shot.vx else 3)
        for shot in self.shots
      ):
        self.multiplier = min(99, self.multiplier + 1)
        reward += self.multiplier
      else:
        self.terminal |= abs(enemy.x - 50) < 5 and abs(enemy.y - self.y) < 5
        if -4 <= enemy.y <= 104:
          survivors.append(enemy)
    self.enemies = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row = int(self.y) // 10
    state[row, 5, 0] = True
    state[row, 5, 4] = self.direction > 0
    for shot in self.shots:
      if 0 <= shot.x < 100 and 0 <= shot.y < 100:
        state[int(shot.y) // 10, int(shot.x) // 10, 3] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 100 and 0 <= enemy.y < 100:
        state[int(enemy.y) // 10, int(enemy.x) // 10, 1 if enemy.vx else 2] = True
    for row, value in enumerate((self.steps, self.hold, self.multiplier)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
