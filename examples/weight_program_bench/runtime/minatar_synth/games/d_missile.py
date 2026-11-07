from dataclasses import dataclass
import math

import numpy as np


LAUNCH_ANGLE = -8 * math.pi / 9


@dataclass
class Missile:
  x: float = 60
  y: float = 90
  angle: float = LAUNCH_ANGLE
  speed: float = 0.5


@dataclass
class Enemy:
  x: float
  y: float
  vx: float
  vy: float
  timer: int


class DMissileEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.tap_timer = self.spawn_timer = 0
    self.pressed = self.terminal = False
    self.multiplier = 1.0
    self.missiles: list[Missile] = []
    self.enemies: list[Enemy] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      count = len(self.missiles)
      self.tap_timer = 99 if count == 0 else 9 if count < 3 else 0
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance_frame(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _enemy(self, x: float, y: float, timer: int) -> Enemy:
    angle = math.atan2(90 - y, self.random.uniform(10, 90) - x)
    return Enemy(x, y, math.cos(angle) * 0.1, math.sin(angle) * 0.1, timer)

  def _advance_frame(self, released: bool) -> int:
    self.tap_timer = max(0, self.tap_timer - 1)
    if released and self.tap_timer:
      self.missiles.append(Missile())
      self.multiplier = max(1, self.multiplier * 0.9)
    for missile in self.missiles:
      missile.x += math.cos(missile.angle) * missile.speed
      missile.y += math.sin(missile.angle) * missile.speed
      missile.speed *= 1.002
      if self.pressed:
        missile.angle = (missile.angle + missile.speed * (0.1 if missile.speed < 3 else 0.01)) % math.tau
    self.missiles = [missile for missile in self.missiles
                     if -3 <= missile.x < 103 and -3 <= missile.y < 90]
    self.spawn_timer = self.spawn_timer - 1 if self.enemies else -1
    if self.spawn_timer < 0:
      count = int(self.random.integers(3, 6))
      for _ in range(count):
        self.enemies.append(self._enemy(self.random.uniform(10, 90), -self.random.uniform(5, 25),
                                        int(self.random.integers(300, 401))))
      self.spawn_timer = 150 * count

    reward = 0
    survivors = []
    for enemy in self.enemies:
      enemy.x += enemy.vx
      enemy.y += enemy.vy
      enemy.timer -= 1
      if enemy.timer == 0:
        for _ in range(int(self.random.integers(2, 5))):
          survivors.append(self._enemy(enemy.x, enemy.y, 999))
      elif any(abs(missile.x - enemy.x) < 3.5 and abs(missile.y - enemy.y) < 3.5
               for missile in self.missiles):
        reward += math.floor(self.multiplier)
        self.multiplier += 1 / len(self.missiles)
      else:
        survivors.append(enemy)
      self.terminal |= enemy.y > 90
    self.enemies = survivors
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, :, 4] = True
    state[9, 6, 0] = len(self.missiles) < 3
    for missile in self.missiles:
      if 0 <= missile.x < 100 and 0 <= missile.y < 100:
        state[int(missile.y) // 10, int(missile.x) // 10, 0] = True
      x = missile.x + 8 * math.cos(missile.angle)
      y = missile.y + 8 * math.sin(missile.angle)
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.x < 100 and 0 <= enemy.y < 100:
        row, col = int(enemy.y) // 10, int(enemy.x) // 10
        state[row, col, 2] = True
        state[row, col, 3] |= enemy.timer <= 40
    for row, value in enumerate((self.steps, round(self.multiplier * 8), self.tap_timer, len(self.missiles))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
