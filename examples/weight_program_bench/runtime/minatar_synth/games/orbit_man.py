from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Planet:
  x: float
  y: float
  radius: float


class OrbitManEnvironment:
  score_cap = 120

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.home = Planet(50, 0, 5)
    self.planets = [self.home]
    self.x, self.y = 50.0, 0.0
    self.angle = math.pi / 2
    self.distance = 20.0
    self.direction = self.multiplier = 1
    self.flight = self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
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

  def _aligned(self):
    cosine, sine = math.cos(self.angle), math.sin(self.angle)
    aligned = []
    for planet in self.planets:
      if planet is self.home or planet.y <= -planet.radius - 4:
        continue
      dx, dy = planet.x - self.home.x, planet.y - self.home.y
      along = dx * cosine + dy * sine
      across = -dx * sine + dy * cosine
      end = min(99 + self.home.radius * 1.485, max(self.home.radius * 1.485, along))
      if math.hypot(along - end, across) < planet.radius + 2:
        aligned.append(planet)
    return aligned

  def _advance(self, tap):
    self.flight = max(1, self.flight - 1)
    scroll = 0.05 + max(0, 80 - self.home.y) * 0.1 / self.flight
    self.distance -= scroll
    while self.distance < 0:
      radius = self.random.uniform(4, 9)
      self.planets.append(Planet(self.random.uniform(10 + radius, 90 - radius), self.distance - 30, radius))
      self.distance += radius * self.random.uniform(1, 2)
    self.angle += 0.03 * self.direction
    for planet in self.planets:
      planet.y += scroll
    self.planets = [p for p in self.planets if p is self.home or p.y <= 100 + p.radius * 2]
    reward = 0
    if tap:
      aligned = self._aligned()
      if not aligned:
        self.terminal = True
        return 0
      target = max(aligned, key=lambda p: math.hypot(p.x - self.home.x, p.y - self.home.y))
      destroyed = [self.home, *(p for p in aligned if p is not target)]
      self.multiplier = max(1, self.multiplier - 1)
      reward = len(destroyed) * (2 * self.multiplier + len(destroyed) - 1) // 2
      self.multiplier += len(destroyed)
      self.planets = [p for p in self.planets if all(p is not old for old in destroyed)]
      self.home = target
      self.angle += math.pi
      self.direction *= -1
      self.flight = 20
    self.x += (self.home.x + self.home.radius * math.cos(self.angle) - self.x) * 0.1
    self.y += (self.home.y + self.home.radius * math.sin(self.angle) - self.y) * 0.1
    self.terminal = self.home.y - self.home.radius > 99
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    if 0 <= self.y < 100:
      state[int(self.y // 10), int(self.x // 10), 0] = True
    aligned = self._aligned()
    for planet in self.planets:
      if 0 <= planet.y < 100:
        row, col = int(planet.y // 10), int(planet.x // 10)
        state[row, col, 1 if planet is self.home else 2] = True
        state[row, col, 3] |= planet in aligned
    for distance in range(0, 100, 4):
      x = self.home.x + distance * math.cos(self.angle)
      y = self.home.y + distance * math.sin(self.angle)
      if 0 <= x < 100 and 0 <= y < 100:
        state[int(y // 10), int(x // 10), 4] = True
    values = (self.steps, min(511, self.multiplier), self.flight, round(self.angle % math.tau * 80))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
