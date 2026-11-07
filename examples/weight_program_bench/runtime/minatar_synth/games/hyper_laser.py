from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Projectile:
  x: float
  y: float
  vx: float
  vy: float
  age: int = 0


@dataclass
class Enemy:
  x: float
  y: float
  interval: float
  charge: float
  burst_size: int
  heading: float = math.pi / 2
  target: float = math.pi / 2
  turret: float = math.pi / 2
  turn_timer: float = 0
  burst_timer: int = 0
  remaining: int = 0

  @property
  def shielded(self) -> bool:
    return self.charge > 0 and self.y < 70


class HyperLaserEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 95.0
    self.aim = self.steps = self.action = self.laser_timer = 0
    self.block_timer = self.enemy_timer = 0.0
    self.multiplier = 1
    self.terminal = False
    self.blocks: list[tuple[float, float]] = []
    self.enemies: list[Enemy] = []
    self.lasers: list[Projectile] = []
    self.shots: list[Projectile] = []

  @property
  def angle(self) -> float:
    return -math.pi / 2 + self.aim * math.pi / 12

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    self.aim = min(5, max(-5, self.aim + (self.action == 4) - (self.action == 2)))
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _blocked(self, x: float, y: float) -> bool:
    return any(abs(x - bx) < 4 and abs(y - by) < 4 for bx, by in self.blocks)

  def _advance(self) -> int:
    shielded = self.action != 5
    if shielded:
      self.x = min(93, max(7, self.x + 0.3 * ((self.action == 3) - (self.action == 1))))
    else:
      self.laser_timer -= 1
      if self.laser_timer < 0:
        self.lasers.append(Projectile(self.x, self.y, 3 * math.cos(self.angle), 3 * math.sin(self.angle)))
        self.laser_timer = 9
    self.block_timer -= 0.1
    if self.block_timer < 0:
      x = self.random.uniform(8, 92)
      if not self._blocked(x, -9):
        self.blocks.append((x, -9))
      self.block_timer = self.random.uniform(0, 16)
    self.blocks = [(x, y + 0.1) for x, y in self.blocks
                   if y < 103 and (abs(x - self.x) >= 5 or abs(y - self.y) >= 5)]
    self._move_lasers()
    if not self.enemies:
      self.block_timer = 0
    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      interval = self.random.uniform(60, 120)
      self.enemies.append(Enemy(self.random.uniform(10, 90), -9, interval,
                                self.random.uniform(0, interval), int(self.random.integers(3, 9))))
      self.enemy_timer = self.random.uniform(150, 200)
    reward = self._move_enemies()
    remaining = []
    for shot in self.shots:
      shot.x += shot.vx
      shot.y += shot.vy + 0.1
      if abs(shot.x - self.x) < (6 if shielded else 4) and abs(shot.y - self.y) < (6 if shielded else 4):
        if shielded:
          reward += self.multiplier
          self.multiplier += 1
        else:
          self.terminal = True
      elif 5 <= shot.x <= 95 and -9 <= shot.y <= 100 and not self._blocked(shot.x, shot.y):
        remaining.append(shot)
    self.shots = remaining
    return reward

  def _move_lasers(self) -> None:
    surviving = []
    for laser in self.lasers:
      laser.y += 0.1
      x, y = laser.x + laser.vx, laser.y + laser.vy
      hit_x = not 5 <= x <= 95 or self._blocked(x, laser.y)
      hit_y = self._blocked(laser.x, y)
      corner = not (hit_x or hit_y) and self._blocked(x, y)
      if hit_x or corner:
        laser.vx *= -1
      if hit_y or corner:
        laser.vy *= -1
      laser.x += laser.vx
      laser.y += laser.vy
      laser.age += 1
      if -5 <= laser.y <= 105 and laser.age <= 120 and not self._blocked(laser.x, laser.y):
        surviving.append(laser)
    self.lasers = surviving

  def _move_enemies(self) -> int:
    reward = 0
    remaining = []
    for enemy in self.enemies:
      enemy.y += 0.1
      delta = (enemy.target - enemy.heading + math.pi) % math.tau - math.pi
      if abs(delta) >= 0.1:
        enemy.heading += delta * 0.1
      else:
        enemy.heading = enemy.target
        x = enemy.x + 0.1 * math.cos(enemy.heading)
        y = enemy.y + 0.1 * math.sin(enemy.heading)
        enemy.turn_timer -= 1
        if enemy.turn_timer < 0 or not 9 <= x <= 90 or self._blocked(x, y):
          enemy.target += math.pi / 2 * self.random.choice((-1, 1))
          enemy.turn_timer = self.random.uniform(200, 300)
        else:
          enemy.x, enemy.y = x, y
      enemy.charge -= 1
      shielded = enemy.shielded
      if shielded:
        angle = math.atan2(self.y - enemy.y, self.x - enemy.x)
        enemy.turret = round(angle / (math.pi / 4)) * math.pi / 4
      if enemy.charge < 0:
        if not enemy.remaining:
          enemy.remaining = enemy.burst_size
          enemy.burst_timer = 0
        enemy.burst_timer -= 1
        if enemy.burst_timer < 0:
          self.shots.append(Projectile(enemy.x, enemy.y, 2 * math.cos(enemy.turret),
                                       2 * math.sin(enemy.turret)))
          enemy.remaining -= 1
          enemy.burst_timer = 20
          if not enemy.remaining:
            enemy.charge = enemy.interval
      hit = [laser for laser in self.lasers if abs(laser.x - enemy.x) < 5 and abs(laser.y - enemy.y) < 5]
      if hit:
        self.lasers = [laser for laser in self.lasers if laser not in hit]
      if hit and not shielded:
        reward += self.multiplier * 10
      else:
        self.terminal |= enemy.y > 99
        remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, int(self.x) // 10, 0] = True
    for x, y in self.blocks:
      if 0 <= y < 100:
        state[int(y) // 10, int(x) // 10, 1] = True
    for enemy in self.enemies:
      if 0 <= enemy.y < 100:
        row, col = int(enemy.y) // 10, int(enemy.x) // 10
        state[row, col, 2] = True
        state[row, col, 3] |= enemy.shielded
    for channel, projectiles in ((4, self.shots), (5, self.lasers)):
      for p in projectiles:
        if 0 <= p.x < 100 and 0 <= p.y < 100:
          state[int(p.y) // 10, int(p.x) // 10, channel] = True
    for row, value in enumerate((self.aim + 5, min(self.multiplier, 31))):
      for bit in range(5):
        state[bit, 0 if row == 0 else 9, 0] = bool(value & (1 << bit))
    state[5, 9, 0] = self.action != 5
    return state
