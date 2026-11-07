from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Moving:
  x: float
  y: float
  angle: float

  def move(self, distance: float) -> None:
    self.x += math.cos(self.angle) * distance
    self.y += math.sin(self.angle) * distance

  def visible(self, margin: float = 0) -> bool:
    return -margin <= self.x < 100 + margin and -margin <= self.y < 100 + margin


@dataclass
class Enemy(Moving):
  aim: float
  windup: int = 0


class ClockTurretEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.fire_timer = self.enemy_timer = self.bullet_timer = 0
    self.angle = -math.pi / 2
    self.direction = self.multiplier = 1
    self.pressed = self.terminal = False
    self.enemies: list[Enemy] = []
    self.shots: list[Moving] = []
    self.bullets: list[Moving] = []
    self.coins: list[Moving] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction *= -1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _near_hand(self, x: float, y: float, radius: float) -> bool:
    dx, dy = x - 50, y - 50
    cos, sin = math.cos(self.angle), math.sin(self.angle)
    along = min(14, max(-6, dx * cos + dy * sin))
    return math.hypot(dx - along * cos, dy - along * sin) < radius

  def _advance_frame(self) -> int:
    for shot in self.shots:
      shot.move(3)
    self.shots = [shot for shot in self.shots if shot.visible(5)]
    if self.pressed:
      self.fire_timer += 1
      if self.fire_timer > 7:
        self.fire_timer -= 7
        for angle, distance in ((self.angle, 12), (self.angle + math.pi, 5)):
          shot = Moving(50, 50, angle)
          shot.move(distance)
          self.shots.append(shot)
    else:
      self.angle = (self.angle + self.direction * 0.1) % math.tau

    coins = []
    for coin in self.coins:
      coin.move(1)
      if self._near_hand(coin.x, coin.y, 5):
        self.multiplier += 1
      elif coin.visible(5):
        coins.append(coin)
    self.coins = coins
    self._spawn()

    reward = 0
    enemies = []
    for enemy in self.enemies:
      enemy.move(0.5)
      if enemy.windup:
        enemy.windup -= 1
        if not enemy.windup:
          self.bullets.append(Moving(enemy.x, enemy.y, enemy.aim))
      if any(math.hypot(enemy.x - shot.x, enemy.y - shot.y) < 5 for shot in self.shots):
        reward += self.multiplier
        self.coins.append(Moving(enemy.x, enemy.y, enemy.aim))
      elif (enemy.x - 50) * math.cos(enemy.angle) + (enemy.y - 50) * math.sin(enemy.angle) < 75:
        enemies.append(enemy)
    self.enemies = enemies

    for bullet in self.bullets:
      for _ in range(3):
        bullet.move(3)
        self.terminal |= self._near_hand(bullet.x, bullet.y, 3.5)
    self.bullets = [bullet for bullet in self.bullets if bullet.visible(5)]
    return reward

  def _spawn(self) -> None:
    self.bullet_timer -= 1
    if self.bullet_timer < 0 and self.enemies:
      enemy = self.enemies[int(self.random.integers(len(self.enemies)))]
      predicted = Moving(enemy.x, enemy.y, enemy.angle)
      predicted.move(22.5)
      angle = math.atan2(50 - predicted.y, 50 - predicted.x)
      offset = (angle - enemy.aim + math.pi) % math.tau - math.pi
      if not enemy.windup and predicted.visible(-5) and abs(offset) > math.pi * 0.07:
        enemy.windup = 45
        self.bullet_timer = int(self.random.integers(45, 61))
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      angle = self.random.uniform(0, math.tau)
      aim = angle + self.random.choice((-1, 1)) * math.pi / 2
      enemy = Enemy(50, 50, angle, aim)
      enemy.move(-60)
      distance = self.random.uniform(30, 40)
      enemy.x -= distance * math.cos(aim)
      enemy.y -= distance * math.sin(aim)
      self.enemies.append(enemy)
      self.enemy_timer = int(self.random.integers(50, 91))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for distance in range(-6, 15, 2):
      point = Moving(50, 50, self.angle)
      point.move(distance)
      state[int(point.y) // 10, int(point.x) // 10, 0] = True
    for channel, objects in enumerate((self.shots, self.enemies, self.bullets, self.coins), 1):
      for obj in objects:
        if obj.visible():
          state[int(obj.y) // 10, int(obj.x) // 10, channel] = True
    for enemy in self.enemies:
      if enemy.windup:
        warning = Moving(enemy.x, enemy.y, enemy.aim)
        for _ in range(20):
          if warning.visible():
            state[int(warning.y) // 10, int(warning.x) // 10, 3] = True
          warning.move(5)
    for row, value in enumerate((self.steps, self.multiplier, round(self.angle % math.tau * 80), self.fire_timer)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
