from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Orb:
  x: float
  y: float
  life: int = 300
  phase: float = 0.0


@dataclass
class Obstacle:
  x: float
  y: float
  vx: float


class InkinhEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 50.0, 80.0
    self.vx = self.vy = 0.0
    self.angle = -math.pi / 2
    self.stretch = 1.0
    self.combo = self.combo_timer = self.steps = self.frames = 0
    self.pressed = self.terminal = False
    self.orbs: list[Orb] = []
    self.obstacles: list[Obstacle] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    self.combo_timer = max(0, self.combo_timer - 1)
    if not self.combo_timer:
      self.combo = 0
    if self.frames % 72 == 0:
      self.obstacles.append(Obstacle(self.random.uniform(10, 90), -8, self.random.uniform(-0.2, 0.2)))
    if self.frames % 30 == 0 and len(self.orbs) < 8:
      for _ in range(9):
        x, y = self.random.uniform(15, 85), self.random.uniform(20, 90)
        if math.hypot(x - self.x, y - self.y) > 20:
          break
      self.orbs.append(Orb(x, y, phase=self.random.uniform(0, math.tau)))
    self.frames += 1
    self.angle += 0.02 if self.pressed else 0.05
    if self.pressed:
      self.vx += 0.15 * math.cos(self.angle)
      self.vy += 0.15 * math.sin(self.angle)
      for orb in self.orbs:
        dx, dy = self.x - orb.x, self.y - orb.y
        distance = math.hypot(dx, dy)
        if 0 < distance < 30:
          orb.x += 0.6 * dx / distance
          orb.y += 0.6 * dy / distance
    self.vx *= 0.96
    self.vy *= 0.96
    self.x = min(95, max(5, self.x + self.vx))
    self.y = min(95, max(5, self.y + self.vy))
    self.stretch += (1 + math.hypot(self.vx, self.vy) * 0.3 - self.stretch) * 0.2
    reward = 0
    remaining = []
    for orb in self.orbs:
      orb.life -= 1
      orb.phase += 0.1
      size = (4 + 2 * min(1, orb.life / 60)) * (1 + 0.15 * math.sin(orb.phase))
      if self._hits(orb.x, orb.y, size):
        self.combo += 1
        self.combo_timer = 60
        reward += self.combo
      elif orb.life:
        remaining.append(orb)
    self.orbs = remaining
    for obstacle in self.obstacles:
      obstacle.x += obstacle.vx
      obstacle.y += 0.5
      self.terminal |= self._hits(obstacle.x, obstacle.y, 7)
    self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.y < 110]
    return reward

  def _hits(self, x: float, y: float, size: float) -> bool:
    return (abs(self.x - x) < size / 2 + 3 / self.stretch
            and abs(self.y - y) < size / 2 + 3 * self.stretch)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y) // 10, int(self.x) // 10, 0] = True
    for orb in self.orbs:
      state[int(orb.y) // 10, int(orb.x) // 10, 1] = True
    for obstacle in self.obstacles:
      if 0 <= obstacle.x < 100 and 0 <= obstacle.y < 100:
        state[int(obstacle.y) // 10, int(obstacle.x) // 10, 2] = True
    x, y = self.x + 10 * math.cos(self.angle), self.y + 10 * math.sin(self.angle)
    if 0 <= x < 100 and 0 <= y < 100:
      state[int(y) // 10, int(x) // 10, 3] = True
    if self.pressed:
      rows, cols = np.ogrid[5:100:10, 5:100:10]
      state[:, :, 4] = (cols - self.x) ** 2 + (rows - self.y) ** 2 < 30 ** 2
    gauges = (self.steps, self.combo, self.combo_timer, round(self.angle % math.tau * 80),
              round((self.vx + 4) * 60), round((self.vy + 4) * 60))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
