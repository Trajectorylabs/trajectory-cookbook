from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Flower:
  x: float
  y: float
  size: float
  pollinated: bool = False
  glow: float = 0


class NeonPollinatorEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.energy, self.glow = 50.0, 100.0, 5.0
    self.flower_distance = self.orb_distance = 0.0
    self.steps = 0
    self.multiplier = 1
    self.pressed = self.terminal = False
    self.flowers: list[Flower] = []
    self.orbs: list[tuple[float, float]] = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    if pressed and not self.pressed:
      self.multiplier = 1
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    if self.pressed and self.energy > 0:
      self.y -= 1
      self.glow = min(25, self.glow + 0.25)
      self.energy -= 0.5
    else:
      self.y += 0.7
      self.glow = max(3, self.glow - 0.5)
      self.energy -= 0.1
    self.energy = max(0, self.energy)
    if not 0 <= self.y <= 100:
      self.terminal = True
      return 0
    self.flower_distance -= 0.5
    if self.flower_distance < 0:
      self.flowers.append(Flower(115, self.random.uniform(10, 90), self.random.uniform(5, 15)))
      self.flower_distance += 20
    for flower in self.flowers:
      flower.x -= 0.5
    self.flowers = [flower for flower in self.flowers if flower.x >= -10]
    lights = [(f.x, f.y, f.glow) for f in self.flowers if f.glow > 0]
    if self.glow > 3:
      lights.append((20, self.y, self.glow))
    reward = 0
    for flower in self.flowers:
      if not flower.pollinated and any(
          abs(radius - flower.size) - 2 < math.hypot(x - flower.x, y - flower.y)
          < radius + flower.size + 2 for x, y, radius in lights):
        flower.pollinated = True
        flower.glow = flower.size * 3
        reward += self.multiplier
        self.multiplier += 1
      flower.glow = max(0, flower.glow - 0.2)
    self.orb_distance -= 0.5
    if self.orb_distance < 0:
      self.orbs.append((100, self.random.uniform(10, 90)))
      self.orb_distance += self.random.uniform(16, 20)
    remaining = []
    for x, y in self.orbs:
      if abs(x - 20) < 7 and abs(y - self.y) < 7:
        self.energy = min(100, self.energy + 20)
      elif x >= -9.5:
        remaining.append((x - 0.5, y))
    self.orbs = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), 2, 0] = True
    rows, cols = np.ogrid[5:100:10, 5:100:10]
    if self.glow > 3:
      state[:, :, 3] = abs(np.hypot(cols - 20, rows - self.y) - self.glow) < 6
    for flower in self.flowers:
      distance = np.hypot(cols - flower.x, rows - flower.y)
      state[:, :, 2 if flower.pollinated else 1] |= abs(distance - flower.size) < 6
      if flower.glow:
        state[:, :, 3] |= abs(distance - flower.glow) < 6
    for x, y in self.orbs:
      visible = math.hypot(x - 20, y - self.y) < self.glow or any(
        math.hypot(x - f.x, y - f.y) < f.glow for f in self.flowers
      )
      if visible and 0 <= x < 100:
        state[int(y // 10), int(x // 10), 4] = True
    for row, value in enumerate((self.steps, round(self.energy * 4), round(self.glow * 10), self.multiplier)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
