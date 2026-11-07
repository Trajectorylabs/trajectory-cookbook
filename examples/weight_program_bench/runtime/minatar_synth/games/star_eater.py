from dataclasses import dataclass
from math import hypot, sin

import numpy as np


@dataclass
class Asteroid:
  x: float
  y: float
  vx: float
  vy: float
  size: float


class StarEaterEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x = self.y = 50.
    self.charge = self.timer = self.steps = self.frame = 0
    self.blast_timer = self.destroyed = 0
    self.blast_x = self.blast_y = self.blast_radius = 0.
    self.held = self.terminal = False
    self.asteroids = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    release = self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self):
    edge = self.random.integers(4)
    offset = self.random.uniform(0, 100)
    drift = self.random.uniform(-0.3, 0.3)
    speed = self.random.uniform(0.5, 0.7)
    x, y, vx, vy = ((offset, 0, drift, speed), (offset, 100, drift, -speed),
                    (0, offset, speed, drift), (100, offset, -speed, drift))[edge]
    if min(hypot(x - self.x, y - self.y), hypot(x - self.x + 100, y - self.y)) > 50:
      self.asteroids.append(Asteroid(x, y, vx, vy, self.random.uniform(5, 9)))
      self.timer = self.random.uniform(22, 55)

  def _advance(self, release):
    self.timer -= 1
    if self.timer <= 0:
      self._spawn()
    speed = 1.8 if self.held else 1
    self.x += 0.2 * speed
    if self.x > 100:
      self.x = 0.
    self.y = min(95, max(5, self.y + sin(self.frame * 0.02 * speed) * 0.1))
    if self.held:
      self.charge = min(60, self.charge + 1)
    elif release and self.charge > 10:
      self.blast_x, self.blast_y = self.x, self.y
      self.blast_radius = 15 + self.charge * 0.8
      self.blast_timer = 12
      self.charge = self.destroyed = 0
    else:
      self.charge = max(0, self.charge - 2)
    self.blast_timer = max(0, self.blast_timer - 1)
    reward = 0
    remaining = []
    for asteroid in self.asteroids:
      asteroid.x += asteroid.vx
      asteroid.y += asteroid.vy
      if not (-10 < asteroid.x < 110 and -10 < asteroid.y < 110):
        continue
      if self.blast_timer and min(
        hypot(asteroid.x - self.blast_x, asteroid.y - self.blast_y),
        hypot(asteroid.x - self.blast_x + 100, asteroid.y - self.blast_y),
      ) < self.blast_radius + asteroid.size / 2:
        self.destroyed += 1
        reward += self.destroyed ** 2
      else:
        self.terminal |= hypot(asteroid.x - self.x, asteroid.y - self.y) < (
          5 + self.charge * 0.15 + asteroid.size / 2)
        remaining.append(asteroid)
    self.asteroids = remaining
    self.frame += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), min(9, int(self.x // 10)), 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    for asteroid in self.asteroids:
      state[:, :, 1] |= np.hypot(cols - asteroid.x, rows - asteroid.y) < asteroid.size / 2 + 5
    if self.charge:
      state[:, :, 2] = np.hypot(cols - self.x, rows - self.y) < 10 + self.charge * 0.15
    if self.blast_timer:
      for x in (self.blast_x, self.blast_x - 100):
        state[:, :, 3] |= np.hypot(cols - x, rows - self.blast_y) < self.blast_radius
    if self.charge > 10:
      for x in (self.x, self.x - 100):
        state[:, :, 4] |= np.hypot(cols - x, rows - self.y) < 15 + self.charge * 0.8
    values = (self.steps, self.charge, self.held, self.blast_timer, self.destroyed)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
