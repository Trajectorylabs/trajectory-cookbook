from dataclasses import dataclass
import math

import numpy as np


RADIUS = 45
HORIZON = 400


@dataclass
class Fish:
  x: float
  y: float
  angle: float
  speed: float


class FeedingFrenzyEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.spawn_timer = 0.0
    self.multiplier = 1
    self.x, self.y, self.angle = 95.0, 50.0, 0.0
    self.pressed = self.darting = self.terminal = False
    self.fish: list[Fish] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
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

  def _advance(self, tap: bool) -> int:
    if self.darting:
      if self.pressed:
        self.angle = (self.angle - 0.1) % math.tau
      self.x += 3.9 * math.cos(self.angle)
      self.y += 3.9 * math.sin(self.angle)
      distance = math.hypot(self.x - 50, self.y - 50)
      if distance > RADIUS:
        self.darting = False
        self.x = 50 + (self.x - 50) * RADIUS / distance
        self.y = 50 + (self.y - 50) * RADIUS / distance
    else:
      radial = math.atan2(self.y - 50, self.x - 50) - 0.7 / RADIUS
      self.x = 50 + RADIUS * math.cos(radial)
      self.y = 50 + RADIUS * math.sin(radial)
      self.angle = (radial - math.pi / 2) % math.tau
      if tap:
        self.darting = True
        self.angle = (self.angle - 0.5) % math.tau
        self.multiplier = 1

    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      angle = self.random.uniform(0, math.tau)
      radius = self.random.uniform(0, 36)
      self.fish.append(Fish(50 + radius * math.cos(angle), 50 + radius * math.sin(angle),
                            angle + math.pi + self.random.uniform(-1, 1),
                            self.random.uniform(0.03, 0.05)))
      self.spawn_timer = 10 * math.sqrt(len(self.fish))
    reward = 0
    remaining = []
    for fish in self.fish:
      speed = fish.speed * (2.5 if self.darting else 1)
      if math.hypot(fish.x - 50, fish.y - 50) > RADIUS * 1.1:
        speed *= 4
      fish.x += speed * math.cos(fish.angle)
      fish.y += speed * math.sin(fish.angle)
      if math.hypot(fish.x - self.x, fish.y - self.y) < 6:
        reward += self.multiplier
        self.multiplier += 1
      elif 0 <= fish.x < 100 and 0 <= fish.y < 100:
        remaining.append(fish)
      else:
        self.terminal = True
    self.fish = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    row, col = int(self.y) // 10, int(self.x) // 10
    state[row, col, 0] = True
    state[row, col, 4] = self.darting
    for fish in self.fish:
      state[int(fish.y) // 10, int(fish.x) // 10, 1] = True
    for row in range(10):
      for col in range(10):
        state[row, col, 2] = abs(math.hypot(10 * col - 45, 10 * row - 45) - RADIUS) < 6
    x = min(99, max(0, self.x + 8 * math.cos(self.angle)))
    y = min(99, max(0, self.y + 8 * math.sin(self.angle)))
    state[int(y) // 10, int(x) // 10, 3] = True
    for row, value in enumerate((self.steps, self.multiplier, round(self.angle * 80))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
