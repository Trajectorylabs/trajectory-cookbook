from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Wake:
  x: float
  y: float
  vx: float
  life: int = 24


@dataclass
class Barrier:
  x: float
  lane: int
  phase: float
  rotation: float = 0

  @property
  def y(self) -> int:
    return 25 + 25 * self.lane


class JetStreamEnvironment:
  score_cap = 60

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 25.0
    self.vx = self.vy = self.tap_impulse = 0.0
    self.lane_step = self.frames = self.steps = self.spawn_timer = 0
    self.spawn_interval = 120.0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.wake: list[Wake] = []
    self.barriers: list[Barrier] = []

  @property
  def lane(self) -> int:
    return (0, 1, 2, 1)[self.lane_step]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.lane_step = (self.lane_step + 1) % 4
      self.vx += self.tap_impulse
      self.vy -= 1
      self.tap_impulse += 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.frames += 1
    self.tap_impulse *= 0.9
    self.y += (21 + self.lane * 25 - self.y) * 0.1 + self.vy
    drift = 1 if self.lane == 1 else -1
    self.x += self.vx + 0.5 * drift
    self.vx *= 0.9
    self.vy *= 0.9
    if self.frames % 3 == 0:
      self.wake.append(Wake(self.x - drift * 5, self.y + 4, -drift * 0.3))
    for particle in self.wake:
      particle.x += particle.vx
      particle.y -= 0.1
      particle.life -= 1
    self.wake = [p for p in self.wake if p.life and -5 <= p.x <= 105]
    self.spawn_timer += 1
    if self.spawn_timer >= self.spawn_interval + self.random.uniform(-30, 30):
      lane = int(self.random.integers(3))
      self.barriers.append(Barrier(110 if lane == 1 else -10, lane, self.random.uniform(0, math.tau)))
      self.spawn_timer = 0
      self.spawn_interval = self.random.uniform(100, 140)
    reward = 0
    remaining = []
    for barrier in self.barriers:
      direction = -1 if barrier.lane == 1 else 1
      barrier.x += direction
      barrier.rotation += 0.15 * direction
      barrier.phase += 0.2
      if self._wake_hit(barrier):
        reward += self.multiplier
        self.multiplier = min(16, self.multiplier + 1)
      elif math.hypot(self.x - barrier.x, self.y - barrier.y) < 6 + 0.9 * math.sin(barrier.phase):
        self.vx += direction * 3
        self.vy += self.random.uniform(-0.5, 0.5)
      elif not -15 <= barrier.x <= 115:
        self.multiplier = max(1, self.multiplier - 1)
      else:
        remaining.append(barrier)
    self.barriers = remaining
    self.terminal = not -3 <= self.x <= 103
    return reward

  def _wake_hit(self, barrier: Barrier) -> bool:
    for blade in range(4):
      angle = barrier.rotation + blade * math.pi / 2
      x = barrier.x + 6 * math.cos(angle)
      y = barrier.y + 6 * math.sin(angle)
      if any(math.hypot(p.x - x, p.y - y) < 3.5 + (24 - p.life) / 24 for p in self.wake):
        return True
    return False

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y) // 10)), min(9, max(0, int(self.x) // 10)), 0] = True
    for particle in self.wake:
      if 0 <= particle.x < 100 and 0 <= particle.y < 100:
        state[int(particle.y) // 10, int(particle.x) // 10, 1] = True
    for barrier in self.barriers:
      if 0 <= barrier.x < 100:
        state[barrier.y // 10, int(barrier.x) // 10, 2 + (barrier.lane == 1)] = True
    state[(21 + self.lane * 25) // 10, :, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.lane_step,
                                 min(511, max(0, round((self.vx + 10) * 20))))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
