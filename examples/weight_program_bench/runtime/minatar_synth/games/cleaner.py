from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Particle:
  x: float
  y: float
  radius: float


class CleanerEnvironment:
  score_cap = 1000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.frame = 0
    self.pressed = self.terminal = False
    self.x = self.head_x = 50.0
    self.head_y = 130.0
    self.length = 0.0
    self.dust_timer, self.hole_timer = 0, 99
    self.multiplier = 1
    self.dust: list[Particle] = []
    self.holes: list[Particle] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.multiplier = 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    angle = math.sin(self.frame * 0.1)
    self.frame += 1
    self.head_x = self.x + math.sin(angle) * self.length
    self.head_y = 130 - math.cos(angle) * self.length
    if self.pressed:
      self.length += (150 - self.length) * 0.005
    else:
      self.x = min(100, max(0, self.x + (self.head_x - self.x) * 0.005 * self.length))
      self.length *= 0.9

    self.dust_timer -= 1
    if self.dust_timer < 0:
      self.dust.append(Particle(self.random.uniform(10, 90), -20, self.random.uniform(1.5, 4.5)))
      self.dust_timer = int(self.random.integers(10, 21))
    reward = 0
    remaining = []
    for dust in self.dust:
      dust.y += 1
      if self._sweeps(dust):
        reward += self.multiplier
        self.multiplier += 1
      elif dust.y <= 170:
        remaining.append(dust)
    self.dust = remaining

    self.hole_timer -= 1
    if self.hole_timer < 0:
      self.holes.append(Particle(self.random.uniform(0, 100), -20, self.random.uniform(5, 12)))
      self.hole_timer = int(self.random.integers(99, 121))
    for hole in self.holes:
      hole.y += 1
      if abs(math.hypot(hole.x - self.x, hole.y - 139) - hole.radius) <= 7:
        self.terminal = True
    self.holes = [hole for hole in self.holes if hole.y <= 170]
    return reward

  def _sweeps(self, dust: Particle) -> bool:
    if abs(dust.x - self.head_x) < 7.5 + dust.radius and abs(dust.y - self.head_y) < 5 + dust.radius:
      return True
    dx, dy = self.head_x - self.x, self.head_y - 130
    px, py = dust.x - self.x, dust.y - 130
    squared = dx * dx + dy * dy
    fraction = min(1, max(0, (px * dx + py * dy) / squared)) if squared else 0
    return math.hypot(px - fraction * dx, py - fraction * dy) < 3.5 + dust.radius

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[9, min(9, int(self.x) // 10), 0] = True
    for fraction in np.linspace(0, 1, 12):
      x = self.x + (self.head_x - self.x) * fraction
      y = 130 + (self.head_y - 130) * fraction
      if 0 <= x <= 100 and 0 <= y < 150:
        state[int(y) // 15, min(9, int(x) // 10), 1] = True
    if 0 <= self.head_x <= 100 and 0 <= self.head_y < 150:
      state[int(self.head_y) // 15, min(9, int(self.head_x) // 10), 4] = True
    for channel, particles in ((2, self.dust), (3, self.holes)):
      for particle in particles:
        if 0 <= particle.y < 150:
          state[int(particle.y) // 15, int(particle.x) // 10, channel] = True
    for row, value in enumerate((self.steps, round(self.length), self.multiplier, self.frame % 63)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
