from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Cloud:
  x: float
  y: float
  vx: float


class KiteMasterEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 35.0, 70.0
    self.vx = self.vy = 0.0
    self.line_length = 20.0
    self.wind = 1.0
    self.steps = self.frames = 0
    self.pressed = self.terminal = False
    self.cloud: Cloud | None = None

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0.0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _advance(self) -> float:
    self.wind = 1 + 0.5 * math.sin(self.frames * 0.01)
    self.frames += 1
    dx, dy = self.x - 20, self.y - 87
    distance = math.hypot(dx, dy)
    if self.pressed:
      self.line_length = max(5, self.line_length - 0.2)
    if distance > self.line_length:
      self.line_length = min(120, self.line_length + self.wind * 0.1)
    tension = max(0, distance - self.line_length) * 0.1
    radial_force = (self.wind * 0.81 - tension) / distance if distance else 0
    fx = dx * radial_force + self.wind * 0.19
    fy = dy * radial_force + 0.3 - self.pressed
    self.vx = (self.vx + fx * 0.02) * 0.98
    self.vy = (self.vy + fy * 0.02) * 0.98
    self.x = min(95, max(5, self.x + self.vx))
    self.y = min(95, max(5, self.y + self.vy))
    if self.cloud is None:
      self.cloud = Cloud(self.random.uniform(25, 75), -5, self.random.uniform(-0.1, 0.1))
    cloud = self.cloud
    cloud.x += cloud.vx
    cloud.y += 0.15
    if (cloud.x <= 20 and cloud.vx < 0) or (cloud.x >= 80 and cloud.vx > 0):
      cloud.vx *= -1
    if cloud.y > 90:
      self.terminal = True
      return 0.0
    reward = 0.0
    if self._collects(cloud):
      reward = 100 - cloud.y
      self.cloud = None
    self.terminal = self.y > 85
    return reward

  def _collects(self, cloud: Cloud) -> bool:
    angle = math.atan2(self.y - 87, self.x - 20) + self.wind * 0.3 + math.pi / 2
    cosine, sine = math.cos(angle), math.sin(angle)
    for offset in (-3, 3):
      cx = cloud.x + offset
      dx, dy = cx - self.x, cloud.y - self.y
      along = min(6, max(-6, dx * cosine + dy * sine))
      if math.hypot(dx - along * cosine, dy - along * sine) < 6:
        return True
      if abs(cx - 20) < 7 and abs(cloud.y - 87) < 7:
        return True
    return False

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, 2, 0] = True
    state[int(self.y) // 10, int(self.x) // 10, 1] = True
    if self.cloud is not None and 0 <= self.cloud.y < 100:
      state[int(self.cloud.y) // 10, int(self.cloud.x) // 10, 2] = True
    for fraction in np.linspace(0, 1, 12):
      x = 20 + fraction * (self.x - 20)
      y = 87 + fraction * (self.y - 87)
      state[int(y) // 10, int(x) // 10, 3] = True
    state[9, :, 4] = True
    values = (self.steps, round(self.line_length * 4), round(self.wind * 100),
              round((self.vx + 3) * 80), round((self.vy + 3) * 80))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
