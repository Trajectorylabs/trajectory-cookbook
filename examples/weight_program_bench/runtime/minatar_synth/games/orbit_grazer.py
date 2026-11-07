from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Asteroid:
  x: float
  y: float
  vx: float
  vy: float
  size: float


class OrbitGrazerEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 75.0, 50.0
    self.radius, self.angle, self.stretch = 25.0, 0.0, 1.0
    self.direction = 1
    self.steps = self.frames = self.shield = 0
    self.pressed = self.terminal = False
    self.asteroids: list[Asteroid] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.direction *= -1
      self.stretch = 1.6
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self):
    angle = self.random.uniform(0, math.tau)
    x, y = 50 + 60 * math.cos(angle), 50 + 60 * math.sin(angle)
    target = self.angle + self.direction * self.random.uniform(0.2, 0.6)
    dx = 50 + self.radius * math.cos(target) - x
    dy = 50 + self.radius * math.sin(target) - y
    if math.hypot(x - self.x, y - self.y) > 50:
      length = math.hypot(dx, dy)
      self.asteroids.append(Asteroid(x, y, 0.5 * dx / length, 0.5 * dy / length,
                                    self.random.uniform(6, 9)))

  def _advance(self):
    self.stretch += (1 - self.stretch) * 0.15
    self.angle += 0.5 * self.direction / (self.radius + 0.1)
    self.x = 50 + self.radius * math.cos(self.angle)
    self.y = 50 + self.radius * math.sin(self.angle)
    if self.frames % 50 == 0:
      self._spawn()
    self.frames += 1
    self.radius = min(40, self.radius + 0.05)
    self.shield = max(-1, self.shield - 1)
    self.terminal = abs(self.x - 50) < 2 + 3 / self.stretch and abs(self.y - 50) < 2 + 3 * self.stretch
    reward = 0
    remaining = []
    shielded_at_start = self.shield > 0
    for asteroid in self.asteroids:
      asteroid.x += asteroid.vx
      asteroid.y += asteroid.vy
      dx, dy = asteroid.x - self.x, asteroid.y - self.y
      if self.shield < 0 and math.hypot(dx, dy) < asteroid.size / 2 + 15:
        reward += 1
      if abs(dx) < asteroid.size / 2 + 3 / self.stretch and abs(dy) < asteroid.size / 2 + 3 * self.stretch:
        if shielded_at_start:
          continue
        self.radius = max(0, self.radius - 20)
        self.shield = 60
      if -10 <= asteroid.x <= 110 and -10 <= asteroid.y <= 110:
        remaining.append(asteroid)
    self.asteroids = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y // 10), int(self.x // 10), 0] = True
    for asteroid in self.asteroids:
      for channel, offset in ((1, 0), (2, 12)):
        x = asteroid.x + offset * asteroid.vx
        y = asteroid.y + offset * asteroid.vy
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y // 10), int(x // 10), channel] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    state[:, :, 3] = abs(np.hypot(cols - 50, rows - 50) - self.radius) < 5
    state[4:6, 4:6, 4] = True
    for row, value in enumerate((self.steps, round(self.radius * 10), self.shield + 1,
                                 round(self.angle % math.tau * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
