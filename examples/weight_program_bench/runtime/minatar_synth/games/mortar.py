from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Enemy:
  x: float
  y: float
  speed: float


@dataclass
class Shot:
  x: float
  y: float
  width: float


@dataclass
class Blast:
  x: float
  y: float
  target_radius: float
  radius: float = 0


class MortarEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.enemies = [Enemy(self.random.uniform(49, 60), self.random.uniform(-149, -140), 0.1)
                    for _ in range(5)]
    self.x, self.direction = 30.0, 1
    self.sight = self.shot = self.blast = None
    self.offset = self.timer = self.max_enemy_y = 0.0
    self.sight_speed = 3
    self.multiplier = 1
    self.steps = self.breach = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap, release = pressed and not self.pressed, self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap, release):
    if self.blast is not None:
      self.blast.radius += (self.blast.target_radius - self.blast.radius) * 0.1
      if self.blast.target_radius - self.blast.radius < 1:
        self.blast = None
    elif self.sight is not None:
      self.sight -= 2 * self.sight_speed
      if self.sight < 0:
        self.offset += (90 - self.sight - self.offset) * 0.05 * self.sight_speed
      if release or self.sight < -200:
        if self.sight >= 0:
          if self.shot is None:
            self.shot = Shot(self.x, 95, math.ceil((91 - self.sight) * 0.2))
        else:
          self.blast = Blast(self.x, self.sight, 2 + min(30, -self.sight * 0.3))
        self.sight = None
        self.sight_speed = max(1, self.sight_speed - 1)
    else:
      self.offset *= 0.8
      self.x += self.direction
      if (self.x < 3 and self.direction < 0) or (self.x > 96 and self.direction > 0):
        self.direction *= -1
      if tap:
        self.sight = 90
        self.multiplier = 1
    if self.shot is not None:
      self.shot.y -= 3
      if self.shot.y < -3:
        self.shot = None
    if self.breach:
      self.offset *= 0.5
      if self.breach > 9:
        self.terminal = True
        return 0
    self.timer = self.timer - 1 if self.enemies else -1
    if self.timer < 0:
      speed = self.random.uniform(0.5, 2) * 0.1
      center = self.random.uniform(20, 80)
      for _ in range(int(self.random.integers(5, 9))):
        x, y = self.random.uniform(-9, 9, 2)
        self.enemies.append(Enemy(center + x, -280 + y, speed * self.random.uniform(0.9, 1.1)))
      self.timer = 99 * math.sqrt(len(self.enemies))
    return self._move_enemies()

  def _move_enemies(self):
    reward = 0
    remaining = []
    rate = math.sqrt(120 - self.max_enemy_y) / 4
    if self.sight is not None:
      rate *= 0.3
    top_progress = -200
    for enemy in self.enemies:
      if enemy.y > 99:
        self.breach += 1
      else:
        shot = self.shot
        blast = self.blast
        hit = shot is not None and abs(shot.x - enemy.x) < shot.width / 2 + 3 and abs(shot.y - enemy.y) < 6
        if blast is not None:
          distance = math.hypot(blast.x - enemy.x, blast.y - enemy.y)
          hit |= abs(distance - blast.radius) < 5.5
        if hit:
          reward += self.multiplier
          self.multiplier += 1
          continue
        enemy.y += enemy.speed * rate
        top_progress = max(top_progress, enemy.y)
      remaining.append(enemy)
    self.enemies = remaining
    self.max_enemy_y = top_progress
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    points = [(self.x, 95, 0)]
    if self.sight is not None:
      points.append((self.x, self.sight, 2))
    if self.shot is not None:
      points.append((self.shot.x, self.shot.y, 3))
    points.extend((e.x, e.y, 1) for e in self.enemies)
    for x, y, channel in points:
      y += self.offset
      if 0 <= y < 100:
        state[int(y // 10), int(x // 10), channel] = True
    if self.blast is not None:
      rows, cols = np.ogrid[5:100:10, 5:100:10]
      state[:, :, 4] = abs(np.hypot(cols - self.blast.x,
                                    rows - self.blast.y - self.offset) - self.blast.radius) < 6
    values = (self.steps, min(511, self.multiplier), self.sight_speed, round(self.offset),
              round(200 + self.sight) if self.sight is not None else 0)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(max(0, value) & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
