from dataclasses import dataclass
from math import cos, hypot, pi, remainder, sin

import numpy as np


@dataclass
class Enemy:
  angle: float
  kind: int
  rate: float
  timer: float
  speed: float


@dataclass
class Shot:
  angle: float
  radius: float
  speed: float


class TrojanDefenseEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle, self.length, self.arc = 0., 10., pi / 8
    self.direction = self.multiplier = 1
    self.enemies, self.shots = [], []
    self.wave_timer = 60.
    self.wave = self.action = self.steps = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    previous = self.action
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    if self.action == 5 and previous != 5:
      self.direction *= -1
      self.multiplier = 1
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.action == 5:
      self.arc = min(pi / 3, self.arc + 0.02)
      self.length = min(20, self.length + 0.5)
      speed_factor = 1 - (self.arc - pi / 8) / (pi / 3 - pi / 8) * 0.7
    else:
      self.arc = max(pi / 8, self.arc - 0.015)
      self.length = max(10, self.length - 0.3)
      speed_factor = 1.2
    self.angle = (self.angle + self.direction * 0.06 * speed_factor) % (2 * pi)
    self.wave_timer -= 1
    if self.wave_timer < 0:
      self._spawn_wave()

    remaining = []
    reward = 0
    for shot in self.shots:
      shot.radius += shot.speed
      if shot.speed < 0:
        offset = abs(remainder(shot.angle - self.angle, 2 * pi))
        if 4 <= shot.radius <= 6.5 + self.length and offset <= self.arc / 2 + 1.5 / shot.radius:
          shot.speed *= -1.5
        elif shot.radius <= 5:
          self.terminal = True
          return reward
      else:
        hit = next((enemy for enemy in self.enemies if _enemy_distance(shot, enemy) <= 4), None)
        if hit is not None:
          self.enemies.remove(hit)
          reward += self.multiplier * (hit.kind + 1) * 10
          self.multiplier += 1
          continue
      if shot.radius <= 55:
        remaining.append(shot)
    self.shots = remaining

    for enemy in self.enemies:
      enemy.timer -= 1
      if enemy.timer <= 0:
        self.shots.append(Shot(enemy.angle, 50, -enemy.speed))
        enemy.timer = enemy.rate * self.random.uniform(0.8, 1.2)
    return reward

  def _spawn_wave(self):
    self.wave += 1
    base = self.random.uniform(0, 2 * pi)
    for _ in range(min(5, 1 + self.wave // 3)):
      angle = base + self.random.uniform(-pi / 4, pi / 4)
      kind = 0 if self.random.random() < 0.7 else int(self.random.integers(1, 3))
      low, high, speed = ((120, 180, 0.5), (80, 120, 0.8), (150, 200, 0.4))[kind]
      self.enemies.append(Enemy(angle, kind, self.random.uniform(low, high),
                                self.random.uniform(30, 60), speed))
    self.wave_timer = self.random.uniform(180, 300)

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    for angle in np.linspace(self.angle - self.arc / 2, self.angle + self.arc / 2, 9):
      for radius in np.linspace(5, 5 + self.length, 9):
        _mark(state, angle, radius, 1)
    for enemy in self.enemies:
      _mark(state, enemy.angle, 50, 2)
    for shot in self.shots:
      _mark(state, shot.angle, shot.radius, 3 if shot.speed < 0 else 4)
    values = (self.steps, self.action, self.direction > 0, round(self.angle * 64),
              round(self.length * 16), round(self.arc * 128), self.multiplier, self.wave)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state


def _enemy_distance(shot, enemy):
  return hypot(shot.radius * cos(shot.angle) - 50 * cos(enemy.angle),
               shot.radius * sin(shot.angle) - 50 * sin(enemy.angle))


def _mark(state, angle, radius, channel):
  x, y = 50 + radius * cos(angle), 50 + radius * sin(angle)
  state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), channel] = True
