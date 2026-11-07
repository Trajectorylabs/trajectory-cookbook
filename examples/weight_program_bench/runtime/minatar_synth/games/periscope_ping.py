from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Contact:
  x: float
  y: float
  heading: float
  triangle: bool = False
  visible: bool = False

  def move(self, speed: float) -> None:
    self.x += math.cos(self.heading) * speed
    self.y += math.sin(self.heading) * speed


class PeriscopePingEnvironment:
  score_cap = 350

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = self.radius = 0.0
    self.direction = 1
    self.steps = self.timer = 0
    self.pressed = self.terminal = False
    self.targets: list[Contact] = []
    self.shots: list[Contact] = []

  @property
  def scope(self):
    return 50 + 10 * math.cos(self.angle), 50 + 10 * math.sin(self.angle)

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

  def _advance(self, tap):
    self.angle = (self.angle + 0.04 * self.direction) % math.tau
    sx, sy = self.scope
    if tap:
      self.radius = 0
      self.direction *= -1
      self.shots.append(Contact(sx, sy, self.angle))
    if self.pressed:
      self.radius += 1
    for shot in self.shots:
      shot.move(1)
    self.shots = [shot for shot in self.shots if math.hypot(shot.x - 50, shot.y - 50) <= 50]
    reward = 0
    remaining = []
    for target in self.targets:
      target.move(0.1)
      distance = math.hypot(target.x - 50, target.y - 50)
      if distance < 5:
        continue
      extent = 2.5 if target.triangle else 3
      if any(abs(shot.x - target.x) < extent + 1.5 and abs(shot.y - target.y) < extent + 1.5
             for shot in self.shots):
        reward += int(distance)
        continue
      scope_hit = abs(target.x - sx) < extent + 1.5 and abs(target.y - sy) < extent + 1.5
      tower_hit = abs(target.x - 50) < extent + 5 and abs(target.y - 50) < extent + 5
      sonar_hit = self.pressed and abs(math.hypot(target.x - sx, target.y - sy) - self.radius) < extent + 2.5
      target.visible = scope_hit or tower_hit or sonar_hit
      self.terminal |= scope_hit or tower_hit
      remaining.append(target)
    self.targets = remaining
    self.timer += 1
    if not self.targets or self.timer >= 200:
      self.timer = 0
      angle = self.random.uniform(0, math.tau)
      triangle = bool(self.random.integers(2))
      heading = angle + math.pi + self.random.uniform(-0.5, 0.5)
      self.targets.append(Contact(50 + 50 * math.cos(angle), 50 + 50 * math.sin(angle), heading, triangle))
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    sx, sy = self.scope
    state[int(sy // 10), int(sx // 10), 0] = True
    for target in self.targets:
      if target.visible and 0 <= target.x < 100 and 0 <= target.y < 100:
        state[int(target.y // 10), int(target.x // 10), 1 + target.triangle] = True
    for shot in self.shots:
      if 0 <= shot.x < 100 and 0 <= shot.y < 100:
        state[int(shot.y // 10), int(shot.x // 10), 3] = True
    if self.pressed:
      rows, cols = np.ogrid[5:100:10, 5:100:10]
      state[:, :, 4] = abs(np.hypot(cols - sx, rows - sy) - self.radius) < 5
    for row, value in enumerate((self.steps, round(self.angle * 80), min(511, int(self.radius)))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.direction > 0
    return state
