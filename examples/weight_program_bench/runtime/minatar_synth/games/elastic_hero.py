from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Platform:
  x: float
  y: float
  width: float


class ElasticHeroEnvironment:
  score_cap = 5000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.stretch = 0
    self.pressed = self.terminal = False
    self.x, self.y, self.vx, self.vy = 50.0, 130.0, 0.0, -1.0
    self.angle, self.water = 0.0, 150.0
    self._generate_platforms()

  def _generate_platforms(self) -> None:
    self.platforms = []
    for y in range(110, 20, -20):
      width = self.random.uniform(20, 30)
      self.platforms.append(Platform(self.random.uniform(width / 2, 100 - width / 2), y, width))

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, released: bool) -> int:
    old_y = self.y
    self.x += self.vx
    self.y += self.vy
    drag = 0.99 if self.y < self.water else 0.9
    self.vx *= drag
    self.vy = (self.vy + 0.1) * drag
    if self.pressed:
      self.stretch = min(20, self.stretch + 1)
      self.angle = (self.angle + 0.1) % math.tau
    elif released:
      self.vx = math.cos(self.angle) * self.stretch * 0.5
      self.vy = math.sin(self.angle) * self.stretch * 0.5
      self.stretch = 0

    if (self.x < 3 and self.vx < 0) or (self.x > 97 and self.vx > 0):
      self.vx *= -0.8
    reward = 0
    if self.y < 3:
      self.y += 150
      reward = math.floor(self.water)
      self.water += 150
      self._generate_platforms()
      old_y = self.y
    if self.y > 147 and self.vy > 0:
      self.vy *= -0.8
    self.water -= 0.1 + max(0, self.water - 140) * 0.05
    for platform in self.platforms:
      if abs(self.x - platform.x) > platform.width / 2 + 5:
        continue
      top, bottom = platform.y - 6.5, platform.y + 6.5
      if self.vy > 0 and old_y <= top <= self.y:
        self.y, self.vy = top, self.vy * -0.8
      elif self.vy < 0 and self.y <= bottom <= old_y:
        self.y, self.vy = bottom, self.vy * -0.8
    self.terminal = self.water < 0
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)

    def mark(x: float, y: float, channel: int) -> None:
      state[min(9, max(0, int(y) // 15)), min(9, max(0, int(x) // 10)), channel] = True

    mark(self.x, self.y, 0)
    for platform in self.platforms:
      for x in np.linspace(platform.x - platform.width / 2, platform.x + platform.width / 2, 9):
        mark(x, platform.y, 1)
    for row in range(10):
      state[row, :, 2] = row * 15 + 7.5 >= self.water
    for radius in (self.stretch / 2, self.stretch):
      mark(self.x + radius * math.cos(self.angle), self.y + radius * math.sin(self.angle), 3)
    state[:, (0, 9), 4] = True
    state[9, :, 4] = True
    for row, value in enumerate((self.steps, self.stretch, round(self.angle * 80),
                                 round(self.water), round(self.vx * 10) + 256, round(self.vy * 10) + 256)):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
