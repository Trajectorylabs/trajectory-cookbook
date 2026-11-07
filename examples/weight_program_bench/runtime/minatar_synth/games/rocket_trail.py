from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np


@dataclass
class Particle:
  x: float
  y: float
  vx: float
  vy: float
  life: int = 0

  def move(self):
    self.x += self.vx
    self.y += self.vy
    self.life -= 1


class RocketTrailEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50., 65.
    self.vx = self.vy = 0.
    self.angle = pi / 2
    self.direction = self.multiplier = 1
    self.steps = self.timer = 0
    self.interval = 60
    self.held = self.terminal = False
    self.flames = []
    self.fuel = []
    self.asteroids = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    self.vx *= 0.95
    self.vy = (self.vy + 0.1) * 0.98
    if self.x <= 45:
      self.direction = 1
    elif self.x >= 55:
      self.direction = -1
    self.angle += (pi / 2 + self.direction - self.angle) * 0.05
    dx, dy = cos(self.angle), sin(self.angle)
    if self.held:
      self.vx -= 0.3 * dx
      self.vy -= 0.3 * dy
      if self.random.random() < 0.3:
        angle = self.angle + self.random.uniform(-0.5, 0.5)
        self.fuel.append(Particle(self.x, self.y, 1.8 * cos(angle), 1.8 * sin(angle), 20))
    self.x += self.vx
    self.y += self.vy
    self.flames.append(Particle(self.x, self.y, 2.1 * dx, 2.1 * dy, 25))
    self.x = min(99, max(1, self.x))
    self.terminal = self.y < 0 or self.y > 105
    for particle in (*self.flames, *self.fuel):
      particle.move()
    self.flames = [particle for particle in self.flames if particle.life > 0]
    self.fuel = [particle for particle in self.fuel if particle.life > 0]
    self.timer += 1
    if not self.asteroids or self.timer >= self.interval:
      x = self.random.uniform(5, 95)
      angle = atan2(104, self.random.uniform(20, 80) - x)
      speed = self.random.uniform(0.3, 0.5)
      self.asteroids.append(Particle(x, -5, speed * cos(angle), speed * sin(angle)))
      self.timer = 0
      self.interval = 80 - self.random.uniform(-30, 30)
    reward = 0
    remaining = []
    for asteroid in self.asteroids:
      asteroid.move()
      if any(self._overlap(asteroid, flame, 3.5 - (25 - flame.life) / 40)
             for flame in self.flames) or any(
               self._overlap(asteroid, fuel, 2.5) for fuel in self.fuel
             ):
        reward += self.multiplier
        self.multiplier = min(16, self.multiplier + 1)
      elif any(abs(asteroid.x - (self.x - offset * dx)) < 3
               and abs(asteroid.y - (self.y - offset * dy)) < 3 for offset in range(6)):
        self.vy += 2
        self.multiplier = 1
      elif asteroid.y > 105:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append(asteroid)
    self.asteroids = remaining
    return reward

  @staticmethod
  def _overlap(asteroid, particle, reach):
    return abs(asteroid.x - particle.x) < reach and abs(asteroid.y - particle.y) < reach

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(self.x // 10), 0] = True
    for channel, particles in ((1, self.flames), (2, self.fuel), (3, self.asteroids)):
      for particle in particles:
        if 0 <= particle.x < 100 and 0 <= particle.y < 100:
          state[int(particle.y // 10), int(particle.x // 10), channel] = True
    x = self.x - 12 * cos(self.angle)
    y = self.y - 12 * sin(self.angle)
    state[int(np.clip(y // 10, 0, 9)), int(np.clip(x // 10, 0, 9)), 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.held,
                                 round(self.angle * 100), round((self.vy + 16) * 16))):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
